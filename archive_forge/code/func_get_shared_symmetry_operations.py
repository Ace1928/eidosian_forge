from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_shared_symmetry_operations(struct: Structure, pointops: list[list[SymmOp]], tol: float=0.1):
    """Get all the point group operations shared by a pair of atomic sites
    in the form [[point operations of site index 1],[],...,[]].

    Args:
        struct: Pymatgen structure
        pointops: list of point group operations from get_site_symmetries method
        tol (float): tolerance to find symmetry operations

    Returns:
        list of lists of shared point operations for each pair of atomic sites
    """
    n_sites = len(struct)
    shared_ops = np.zeros((n_sites, n_sites), dtype=object)
    for site1 in range(n_sites):
        for site2 in range(n_sites):
            shared_ops[site1][site2] = []
            for point_op1 in pointops[site1]:
                for point_op2 in pointops[site2]:
                    if np.allclose(point_op1.rotation_matrix, point_op2.rotation_matrix):
                        shared_ops[site1][site2].append(point_op1)
    for site1, sops in enumerate(shared_ops):
        for site2, sop in enumerate(sops):
            unique_ops = []
            for ops in sop:
                op = SymmOp.from_rotation_and_translation(rotation_matrix=ops.rotation_matrix, translation_vec=(0, 0, 0), tol=tol)
                if op not in unique_ops:
                    unique_ops.append(op)
            shared_ops[site1][site2] = unique_ops
    return shared_ops