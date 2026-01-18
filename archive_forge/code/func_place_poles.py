import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def place_poles(A, B, poles, method='YT', rtol=0.001, maxiter=30):
    """
    Compute K such that eigenvalues (A - dot(B, K))=poles.

    K is the gain matrix such as the plant described by the linear system
    ``AX+BU`` will have its closed-loop poles, i.e the eigenvalues ``A - B*K``,
    as close as possible to those asked for in poles.

    SISO, MISO and MIMO systems are supported.

    Parameters
    ----------
    A, B : ndarray
        State-space representation of linear system ``AX + BU``.
    poles : array_like
        Desired real poles and/or complex conjugates poles.
        Complex poles are only supported with ``method="YT"`` (default).
    method: {'YT', 'KNV0'}, optional
        Which method to choose to find the gain matrix K. One of:

            - 'YT': Yang Tits
            - 'KNV0': Kautsky, Nichols, Van Dooren update method 0

        See References and Notes for details on the algorithms.
    rtol: float, optional
        After each iteration the determinant of the eigenvectors of
        ``A - B*K`` is compared to its previous value, when the relative
        error between these two values becomes lower than `rtol` the algorithm
        stops.  Default is 1e-3.
    maxiter: int, optional
        Maximum number of iterations to compute the gain matrix.
        Default is 30.

    Returns
    -------
    full_state_feedback : Bunch object
        full_state_feedback is composed of:
            gain_matrix : 1-D ndarray
                The closed loop matrix K such as the eigenvalues of ``A-BK``
                are as close as possible to the requested poles.
            computed_poles : 1-D ndarray
                The poles corresponding to ``A-BK`` sorted as first the real
                poles in increasing order, then the complex congugates in
                lexicographic order.
            requested_poles : 1-D ndarray
                The poles the algorithm was asked to place sorted as above,
                they may differ from what was achieved.
            X : 2-D ndarray
                The transfer matrix such as ``X * diag(poles) = (A - B*K)*X``
                (see Notes)
            rtol : float
                The relative tolerance achieved on ``det(X)`` (see Notes).
                `rtol` will be NaN if it is possible to solve the system
                ``diag(poles) = (A - B*K)``, or 0 when the optimization
                algorithms can't do anything i.e when ``B.shape[1] == 1``.
            nb_iter : int
                The number of iterations performed before converging.
                `nb_iter` will be NaN if it is possible to solve the system
                ``diag(poles) = (A - B*K)``, or 0 when the optimization
                algorithms can't do anything i.e when ``B.shape[1] == 1``.

    Notes
    -----
    The Tits and Yang (YT), [2]_ paper is an update of the original Kautsky et
    al. (KNV) paper [1]_.  KNV relies on rank-1 updates to find the transfer
    matrix X such that ``X * diag(poles) = (A - B*K)*X``, whereas YT uses
    rank-2 updates. This yields on average more robust solutions (see [2]_
    pp 21-22), furthermore the YT algorithm supports complex poles whereas KNV
    does not in its original version.  Only update method 0 proposed by KNV has
    been implemented here, hence the name ``'KNV0'``.

    KNV extended to complex poles is used in Matlab's ``place`` function, YT is
    distributed under a non-free licence by Slicot under the name ``robpole``.
    It is unclear and undocumented how KNV0 has been extended to complex poles
    (Tits and Yang claim on page 14 of their paper that their method can not be
    used to extend KNV to complex poles), therefore only YT supports them in
    this implementation.

    As the solution to the problem of pole placement is not unique for MIMO
    systems, both methods start with a tentative transfer matrix which is
    altered in various way to increase its determinant.  Both methods have been
    proven to converge to a stable solution, however depending on the way the
    initial transfer matrix is chosen they will converge to different
    solutions and therefore there is absolutely no guarantee that using
    ``'KNV0'`` will yield results similar to Matlab's or any other
    implementation of these algorithms.

    Using the default method ``'YT'`` should be fine in most cases; ``'KNV0'``
    is only provided because it is needed by ``'YT'`` in some specific cases.
    Furthermore ``'YT'`` gives on average more robust results than ``'KNV0'``
    when ``abs(det(X))`` is used as a robustness indicator.

    [2]_ is available as a technical report on the following URL:
    https://hdl.handle.net/1903/5598

    See Also
    --------
    scipy.signal.place_poles

    References
    ----------
    .. [1] J. Kautsky, N.K. Nichols and P. van Dooren, "Robust pole assignment
           in linear state feedback", International Journal of Control, Vol. 41
           pp. 1129-1155, 1985.
    .. [2] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
           pole assignment by state feedback", IEEE Transactions on Automatic
           Control, Vol. 41, pp. 1432-1452, 1996.
    """
    update_loop, poles = _valid_inputs(A, B, poles, method, rtol, maxiter)
    cur_rtol = 0
    nb_iter = 0
    u, z = cupy.linalg.qr(B, mode='complete')
    rankB = cupy.linalg.matrix_rank(B)
    u0 = u[:, :rankB]
    u1 = u[:, rankB:]
    z = z[:rankB, :]
    if B.shape[0] == rankB:
        diag_poles = cupy.zeros(A.shape)
        idx = 0
        while idx < poles.shape[0]:
            p = poles[idx]
            diag_poles[idx, idx] = cupy.real(p)
            if ~cupy.isreal(p):
                diag_poles[idx, idx + 1] = -cupy.imag(p)
                diag_poles[idx + 1, idx + 1] = cupy.real(p)
                diag_poles[idx + 1, idx] = cupy.imag(p)
                idx += 1
            idx += 1
        gain_matrix = cupy.linalg.lstsq(B, diag_poles - A, rcond=-1)[0]
        transfer_matrix = cupy.eye(A.shape[0])
        cur_rtol = cupy.nan
        nb_iter = cupy.nan
    else:
        ker_pole = []
        skip_conjugate = False
        for j in range(B.shape[0]):
            if skip_conjugate:
                skip_conjugate = False
                continue
            pole_space_j = cupy.dot(u1.T, A - poles[j] * cupy.eye(B.shape[0])).T
            Q, _ = cupy.linalg.qr(pole_space_j, mode='complete')
            ker_pole_j = Q[:, pole_space_j.shape[1]:]
            transfer_matrix_j = cupy.sum(ker_pole_j, axis=1)[:, None]
            transfer_matrix_j = transfer_matrix_j / cupy.linalg.norm(transfer_matrix_j)
            if ~cupy.isreal(poles[j]):
                transfer_matrix_j = cupy.hstack([cupy.real(transfer_matrix_j), cupy.imag(transfer_matrix_j)])
                ker_pole.extend([ker_pole_j, ker_pole_j])
                skip_conjugate = True
            else:
                ker_pole.append(ker_pole_j)
            if j == 0:
                transfer_matrix = transfer_matrix_j
            else:
                transfer_matrix = cupy.hstack((transfer_matrix, transfer_matrix_j))
        if rankB > 1:
            stop, cur_rtol, nb_iter = update_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol)
            if not stop and rtol > 0:
                err_msg = f'Convergence was not reached after maxiter iterations.\nYou asked for a tolerance of {rtol}, we got {cur_rtol}.'
                warnings.warn(err_msg, stacklevel=2)
        transfer_matrix = transfer_matrix.astype(complex)
        idx = 0
        while idx < poles.shape[0] - 1:
            if ~cupy.isreal(poles[idx]):
                rel = transfer_matrix[:, idx].copy()
                img = transfer_matrix[:, idx + 1]
                transfer_matrix[:, idx] = rel - 1j * img
                transfer_matrix[:, idx + 1] = rel + 1j * img
                idx += 1
            idx += 1
        try:
            m = cupy.linalg.solve(transfer_matrix.T, cupy.diag(poles) @ transfer_matrix.T).T
            gain_matrix = cupy.linalg.solve(z, u0.T @ (m - A))
        except cupy.linalg.LinAlgError as e:
            raise ValueError("The poles you've chosen can't be placed. Check the controllability matrix and try another set of poles") from e
    gain_matrix = -gain_matrix
    gain_matrix = cupy.real(gain_matrix)
    full_state_feedback = Bunch()
    full_state_feedback.gain_matrix = gain_matrix
    temp = (A - B @ gain_matrix).get()
    import numpy as np
    poles = np.linalg.eig(temp)[0]
    ordered_poles = _order_complex_poles(cupy.asarray(poles))
    full_state_feedback.computed_poles = ordered_poles
    full_state_feedback.requested_poles = poles
    full_state_feedback.X = transfer_matrix
    full_state_feedback.rtol = cur_rtol
    full_state_feedback.nb_iter = nb_iter
    return full_state_feedback