from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_magnetic_symmetry_dataset(cell: Cell, is_axial=None, symprec=1e-05, angle_tolerance=-1.0, mag_symprec=-1.0) -> dict | None:
    """Search magnetic symmetry dataset from an input cell. If it fails, return None.

    The description of its keys is given at :ref:`magnetic_spglib_dataset`.

    Parameters
    ----------
    cell, is_axial, symprec, angle_tolerance, mag_symprec:
        See :func:`get_magnetic_symmetry`.

    Returns
    -------
    dataset : dict or None
        Dictionary keys are as follows:

        Magnetic space-group type
            - uni_number: int
                UNI number between 1 to 1651
            - msg_type: int
                Magnetic space groups (MSG) is classified by its family space
                group (FSG) and maximal space subgroup (XSG). FSG is a non-magnetic
                space group obtained by ignoring time-reversal term in MSG. XSG is
                a space group obtained by picking out non time-reversal operations
                in MSG.

                - msg_type==1 (type-I):
                    MSG, XSG, FSG are all isomorphic.
                - msg_type==2 (type-II):
                    XSG and FSG are isomorphic, and MSG is generated from XSG and pure time reversal operations.
                - msg_type==3 (type-III):
                    XSG is a proper subgroup of MSG with isomorphic translational subgroups.
                - msg_type==4 (type-IV):
                    XSG is a proper subgroup of MSG with isomorphic point group.

            - hall_number: int
                For type-I, II, III, Hall number of FSG; for type-IV, that of XSG
            - tensor_rank: int

        Magnetic symmetry operations
            - n_operations: int
            - rotations: array, (n_operations, 3, 3)
                Rotation (matrix) parts of symmetry operations
            - translations: array, (n_operations, 3)
                Translation (vector) parts of symmetry operations
            - time_reversals: array, (n_operations, )
                Time reversal part of magnetic symmetry operations.
                True indicates time reversal operation, and False indicates
                an ordinary operation.

        Equivalent atoms
            - n_atoms: int
            - equivalent_atoms: array
                See the docstring of get_symmetry_dataset

        Transformation to standardized setting
            - transformation_matrix: array, (3, 3)
                Transformation matrix from input lattice to standardized
            - origin_shift: array, (3, )
                Origin shift from standardized to input origin

        Standardized crystal structure
            - n_std_atoms: int
                Number of atoms in standardized unit cell
            - std_lattice: array, (3, 3)
                Row-wise lattice vectors
            - std_types: array, (n_std_atoms, )
            - std_positions: array, (n_std_atoms, 3)
            - std_tensors: array
                (n_std_atoms, ) for collinear magnetic moments.
                (n_std_atoms, 3) for vector non-collinear magnetic moments.
            - std_rotation_matrix
                Rigid rotation matrix to rotate from standardized basis
                vectors to idealized standardized basis vectors

        Intermediate data in symmetry search
            - primitive_lattice: array, (3, 3)

    Notes
    -----
    .. versionadded:: 2.0

    """
    _set_no_error()
    lattice, positions, numbers, magmoms = _expand_cell(cell)
    tensor_rank = magmoms.ndim - 1
    if is_axial is None:
        if tensor_rank == 0:
            is_axial = False
        elif tensor_rank == 1:
            is_axial = True
    spg_ds = _spglib.magnetic_dataset(lattice, positions, numbers, magmoms, tensor_rank, is_axial, symprec, angle_tolerance, mag_symprec)
    if spg_ds is None:
        _set_error_message()
        return None
    keys = ('uni_number', 'msg_type', 'hall_number', 'tensor_rank', 'n_operations', 'rotations', 'translations', 'time_reversals', 'n_atoms', 'equivalent_atoms', 'transformation_matrix', 'origin_shift', 'n_std_atoms', 'std_lattice', 'std_types', 'std_positions', 'std_tensors', 'std_rotation_matrix', 'primitive_lattice')
    dataset = {}
    for key, data in zip(keys, spg_ds):
        dataset[key] = data
    dataset['rotations'] = np.array(dataset['rotations'], dtype='intc', order='C')
    dataset['translations'] = np.array(dataset['translations'], dtype='double', order='C')
    dataset['time_reversals'] = np.array(dataset['time_reversals'], dtype='intc', order='C') == 1
    dataset['equivalent_atoms'] = np.array(dataset['equivalent_atoms'], dtype='intc')
    dataset['transformation_matrix'] = np.array(dataset['transformation_matrix'], dtype='double', order='C')
    dataset['origin_shift'] = np.array(dataset['origin_shift'], dtype='double')
    dataset['std_lattice'] = np.array(np.transpose(dataset['std_lattice']), dtype='double', order='C')
    dataset['std_types'] = np.array(dataset['std_types'], dtype='intc')
    dataset['std_positions'] = np.array(dataset['std_positions'], dtype='double', order='C')
    dataset['std_rotation_matrix'] = np.array(dataset['std_rotation_matrix'], dtype='double', order='C')
    dataset['primitive_lattice'] = np.array(np.transpose(dataset['primitive_lattice']), dtype='double', order='C')
    dataset['std_tensors'] = np.array(dataset['std_tensors'], dtype='double', order='C')
    if tensor_rank == 1:
        dataset['std_tensors'] = dataset['std_tensors'].reshape(-1, 3)
    _set_error_message()
    return dataset