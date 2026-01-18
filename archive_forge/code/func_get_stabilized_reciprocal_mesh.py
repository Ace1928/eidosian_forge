from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_stabilized_reciprocal_mesh(mesh, rotations, is_shift=None, is_time_reversal=True, qpoints=None, is_dense=False):
    """Return k-point map to the irreducible k-points and k-point grid points.

    The symmetry is searched from the input rotation matrices in real space.

    Parameters
    ----------
    mesh : array_like
        Uniform sampling mesh numbers.
        dtype='intc', shape=(3,)
    rotations : array_like
        Rotation matrices with respect to real space basis vectors.
        dtype='intc', shape=(rotations, 3)
    is_shift : array_like
        [0, 0, 0] gives Gamma center mesh and value 1 gives  half mesh shift.
        dtype='intc', shape=(3,)
    is_time_reversal : bool
        Time reversal symmetry is included or not.
    qpoints : array_like
        q-points used as stabilizer(s) given in reciprocal space with respect
        to reciprocal basis vectors.
        dtype='double', shape=(qpoints ,3) or (3,)
    is_dense : bool, optional
        grid_mapping_table is returned with dtype='uintp' if True. Otherwise
        its dtype='intc'. Default is False.

    Returns
    -------
    grid_mapping_table : ndarray
        Grid point mapping table to ir-gird-points.
        dtype='intc', shape=(prod(mesh),)
    grid_address : ndarray
        Address of all grid points. Each address is given by three unsigned
        integers.
        dtype='intc', shape=(prod(mesh), 3)

    """
    _set_no_error()
    if is_dense:
        dtype = 'uintp'
    else:
        dtype = 'intc'
    mapping_table = np.zeros(np.prod(mesh), dtype=dtype)
    grid_address = np.zeros((np.prod(mesh), 3), dtype='intc')
    if is_shift is None:
        is_shift = [0, 0, 0]
    if qpoints is None:
        qpoints = np.array([[0, 0, 0]], dtype='double', order='C')
    else:
        qpoints = np.array(qpoints, dtype='double', order='C')
        if qpoints.shape == (3,):
            qpoints = np.array([qpoints], dtype='double', order='C')
    if _spglib.stabilized_reciprocal_mesh(grid_address, mapping_table, np.array(mesh, dtype='intc'), np.array(is_shift, dtype='intc'), is_time_reversal * 1, np.array(rotations, dtype='intc', order='C'), qpoints) > 0:
        return (mapping_table, grid_address)
    else:
        return None