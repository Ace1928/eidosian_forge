import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def make_atom14_dists_bounds(overlap_tolerance: float=1.5, bond_length_tolerance_factor: int=15) -> Dict[str, np.ndarray]:
    """compute upper and lower bounds for bonds to assess violations."""
    restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_list = restype_name_to_atom14_names[resname]
        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if not atom2_name or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 10000000000.0
                restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {'lower_bound': restype_atom14_bond_lower_bound, 'upper_bound': restype_atom14_bond_upper_bound, 'stddev': restype_atom14_bond_stddev}