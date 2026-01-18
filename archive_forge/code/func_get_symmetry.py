from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_symmetry(cell: Cell, symprec=1e-05, angle_tolerance=-1.0, mag_symprec=-1.0, is_magnetic=True) -> dict | None:
    """Find symmetry operations from a crystal structure and site tensors.

    .. warning::
        :func:`get_symmetry` with ``is_magnetic=True`` is deprecated at version 2.0.

    Use :func:`get_magnetic_symmetry` for magnetic symmetry search.

    Parameters
    ----------
    cell : tuple
        Crystal structure given in tuple.
        It has to follow the following form,
        (basis vectors, atomic points, types in integer numbers, ...)

        - basis vectors : array_like
            shape=(3, 3), order='C', dtype='double'

            .. code-block:: python

                [[a_x, a_y, a_z],
                [b_x, b_y, b_z],
                [c_x, c_y, c_z]]

        - atomic points : array_like
            shape=(num_atom, 3), order='C', dtype='double'

            Atomic position vectors with respect to basis vectors, i.e.,
            given in  fractional coordinates.
        - types : array_like
            shape=(num_atom, ), dtype='intc'

            Integer numbers to distinguish species.
        - optional data :
            case-I: Scalar
                shape=(num_atom, ), dtype='double'

                Each atomic site has a scalar value. With is_magnetic=True,
                values are included in the symmetry search in a way of
                collinear magnetic moments.
            case-II: Vectors
                shape=(num_atom, 3), order='C', dtype='double'

                Each atomic site has a vector. With is_magnetic=True,
                vectors are included in the symmetry search in a way of
                non-collinear magnetic moments.
    symprec : float
        Symmetry search tolerance in the unit of length.
    angle_tolerance : float
        Symmetry search tolerance in the unit of angle deg.
        Normally it is not recommended to use this argument.
        See a bit more detail at :ref:`variables_angle_tolerance`.
        If the value is negative, an internally optimized routine is used to
        judge symmetry.
    mag_symprec : float
        Tolerance for magnetic symmetry search in the unit of magnetic moments.
        If not specified, use the same value as symprec.
    is_magnetic : bool
        When optional data (4th element of cell tuple) is given in case-II,
        the symmetry search is performed considering magnetic symmetry, which
        may be corresponding to that for non-collinear calculation. Default is
        True, but this does nothing unless optional data is supplied.

    Returns
    -------
    symmetry: dict
        Rotation parts and translation parts of symmetry operations are represented
        with respect to basis vectors.
        When the search failed, :code:`None` is returned.

        - 'rotations' : ndarray
            shape=(num_operations, 3, 3), order='C', dtype='intc'

            Rotation (matrix) parts of symmetry operations
        - 'translations' : ndarray
            shape=(num_operations, 3), dtype='double'

            Translation (vector) parts of symmetry operations
        - 'time_reversals': ndarray (exists when the optional data is given)
            shape=(num_operations, ), dtype='bool\\_'

            Time reversal part of magnetic symmetry operations.
            True indicates time reversal operation, and False indicates
            an ordinary operation.
        - 'equivalent_atoms' : ndarray
            shape=(num_atoms, ), dtype='intc'

            A mapping table of atoms to symmetrically independent atoms.
            This is used to find symmetrically equivalent atoms.
            The numbers contained are the indices of atoms starting from 0,
            i.e., the first atom is numbered as 0, and
            then 1, 2, 3, ... :code:`np.unique(equivalent_atoms)` gives representative
            symmetrically independent atoms. A list of atoms that are
            symmetrically equivalent to some independent atom (here for example 1
            is in :code:`equivalent_atom`) is found by
            :code:`np.where(equivalent_atom=1)[0]`.

    Notes
    -----
    The orders of the rotation matrices and the translation
    vectors correspond with each other, e.g. , the second symmetry
    operation is organized by the set of the second rotation matrix and second
    translation vector in the respective arrays. Therefore a set of
    symmetry operations may obtained by

    .. code-block:: python

        [(r, t) for r, t in zip(dataset['rotations'], dataset['translations'])]

    The operations are given with respect to the fractional coordinates
    (not for Cartesian coordinates). The rotation matrix and translation
    vector are used as follows:

    .. code-block::

        new_vector[3x1] = rotation[3x3] * vector[3x1] + translation[3x1]

    The three values in the vector are given for the a, b, and c axes,
    respectively.

    """
    _set_no_error()
    _, _, _, magmoms = _expand_cell(cell)
    if magmoms is None:
        dataset = get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
        if dataset is None:
            return None
        return {'rotations': dataset['rotations'], 'translations': dataset['translations'], 'equivalent_atoms': dataset['equivalent_atoms']}
    else:
        warnings.warn('Use get_magnetic_symmetry() for cell with magnetic moments.', DeprecationWarning)
        return get_magnetic_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance, mag_symprec=mag_symprec, is_axial=None, with_time_reversal=is_magnetic)