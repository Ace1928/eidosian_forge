from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_BZ_grid_points_by_rotations(address_orig, reciprocal_rotations, mesh, bz_map, is_shift=None, is_dense=False):
    """Return grid points obtained after rotating input grid address.

    Parameters
    ----------
    address_orig : array_like
        Grid point address to be rotated.
        dtype='intc', shape=(3,)
    reciprocal_rotations : array_like
        Rotation matrices {R} with respect to reciprocal basis vectors.
        Defined by q'=Rq.
        dtype='intc', shape=(rotations, 3, 3)
    mesh : array_like
        dtype='intc', shape=(3,)
    bz_map : array_like
        TODO
    is_shift : array_like, optional
        With (1) or without (0) half grid shifts with respect to grid intervals
        sampled along reciprocal basis vectors. Default is None, which
        gives [0, 0, 0].
    is_dense : bool, optional
        rot_grid_points is returned with dtype='uintp' if True. Otherwise
        its dtype='intc'. Default is False.

    Returns
    -------
    rot_grid_points : ndarray
        Grid points obtained after rotating input grid address
        dtype='intc' or 'uintp', shape=(rotations,)

    """
    _set_no_error()
    if is_shift is None:
        _is_shift = np.zeros(3, dtype='intc')
    else:
        _is_shift = np.array(is_shift, dtype='intc')
    if bz_map.dtype == 'uintp' and bz_map.flags.c_contiguous:
        _bz_map = bz_map
    else:
        _bz_map = np.array(bz_map, dtype='uintp')
    rot_grid_points = np.zeros(len(reciprocal_rotations), dtype='uintp')
    _spglib.BZ_grid_points_by_rotations(rot_grid_points, np.array(address_orig, dtype='intc'), np.array(reciprocal_rotations, dtype='intc', order='C'), np.array(mesh, dtype='intc'), _is_shift, _bz_map)
    if is_dense:
        return rot_grid_points
    else:
        return np.array(rot_grid_points, dtype='intc')