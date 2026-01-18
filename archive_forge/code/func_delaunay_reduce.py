from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def delaunay_reduce(lattice, eps=1e-05):
    """Run Delaunay reduction. When the search failed, `None` is returned.

    The transformation from original basis vectors
    :math:`( \\mathbf{a} \\; \\mathbf{b} \\; \\mathbf{c} )`
    to final basis vectors :math:`( \\mathbf{a}' \\; \\mathbf{b}' \\; \\mathbf{c}' )` is achieved by linear
    combination of basis vectors with integer coefficients without
    rotating coordinates. Therefore the transformation matrix is obtained
    by :math:`\\mathbf{P} = ( \\mathbf{a} \\; \\mathbf{b} \\; \\mathbf{c} ) ( \\mathbf{a}' \\; \\mathbf{b}' \\; \\mathbf{c}' )^{-1}` and the matrix
    elements have to be almost integers.

    The algorithm is found in the international tables for crystallography volume A.

    Parameters
    ----------
    lattice: ndarray, (3, 3)
        Lattice parameters in the form of

        .. code-block::

            [[a_x, a_y, a_z],
                [b_x, b_y, b_z],
                [c_x, c_y, c_z]]

    eps: float
        Tolerance parameter, but unlike `symprec` the unit is not a length.
        Tolerance to check if volume is close to zero or not and
        if two basis vectors are orthogonal by the value of dot
        product being close to zero or not.

    Returns
    -------
    delaunay_lattice: ndarray, (3, 3)
        Reduced lattice parameters are given as a numpy 'double' array:

        .. code-block::

            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]

    Notes
    -----
    .. versionadded:: 1.9.4
    """
    _set_no_error()
    delaunay_lattice = np.array(np.transpose(lattice), dtype='double', order='C')
    result = _spglib.delaunay_reduce(delaunay_lattice, float(eps))
    _set_error_message()
    if result == 0:
        return None
    else:
        return np.array(np.transpose(delaunay_lattice), dtype='double', order='C')