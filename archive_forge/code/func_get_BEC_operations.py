from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_BEC_operations(self, eigtol=1e-05, opstol=0.001):
    """
        Returns the symmetry operations which maps the tensors
        belonging to equivalent sites onto each other in the form
        [site index 1, site index 2, [Symmops mapping from site
        index 1 to site index 2]].

        Args:
            eigtol (float): tolerance for determining if two sites are
            related by symmetry
            opstol (float): tolerance for determining if a symmetry
            operation relates two sites

        Returns:
            list of symmetry operations mapping equivalent sites and
            the indexes of those sites.
        """
    bec = self.bec
    struct = self.structure
    ops = SpacegroupAnalyzer(struct).get_symmetry_operations(cartesian=True)
    uniq_point_ops = list(ops)
    for ops in self.pointops:
        for op in ops:
            if op not in uniq_point_ops:
                uniq_point_ops.append(op)
    passed = []
    relations = []
    for site, val in enumerate(bec):
        unique = 1
        eig1, _vecs1 = np.linalg.eig(val)
        index = np.argsort(eig1)
        new_eig = np.real([eig1[index[0]], eig1[index[1]], eig1[index[2]]])
        for index, p in enumerate(passed):
            if np.allclose(new_eig, p[1], atol=eigtol):
                relations.append([site, index])
                unique = 0
                passed.append([site, p[0], new_eig])
                break
        if unique == 1:
            relations.append([site, site])
            passed.append([site, new_eig])
    BEC_operations = []
    for atom, r in enumerate(relations):
        BEC_operations.append(r)
        BEC_operations[atom].append([])
        for op in uniq_point_ops:
            new = op.transform_tensor(self.bec[relations[atom][1]])
            if np.allclose(new, self.bec[r[0]], atol=opstol):
                BEC_operations[atom][2].append(op)
    self.BEC_operations = BEC_operations
    return BEC_operations