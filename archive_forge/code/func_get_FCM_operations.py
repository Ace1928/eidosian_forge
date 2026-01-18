from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_FCM_operations(self, eigtol=1e-05, opstol=1e-05):
    """
        Returns the symmetry operations which maps the tensors
        belonging to equivalent sites onto each other in the form
        [site index 1a, site index 1b, site index 2a, site index 2b,
        [Symmops mapping from site index 1a, 1b to site index 2a, 2b]].

        Args:
            eigtol (float): tolerance for determining if two sites are
            related by symmetry
            opstol (float): tolerance for determining if a symmetry
            operation relates two sites

        Returns:
            list of symmetry operations mapping equivalent sites and
            the indexes of those sites.
        """
    struct = self.structure
    ops = SpacegroupAnalyzer(struct).get_symmetry_operations(cartesian=True)
    uniq_point_ops = list(ops)
    for ops in self.pointops:
        for op in ops:
            if op not in uniq_point_ops:
                uniq_point_ops.append(op)
    passed = []
    relations = []
    for atom1 in range(len(self.fcm)):
        for atom2 in range(atom1, len(self.fcm)):
            unique = 1
            eig1, _vecs1 = np.linalg.eig(self.fcm[atom1][atom2])
            index = np.argsort(eig1)
            new_eig = np.real([eig1[index[0]], eig1[index[1]], eig1[index[2]]])
            for p in passed:
                if np.allclose(new_eig, p[2], atol=eigtol):
                    relations.append([atom1, atom2, p[0], p[1]])
                    unique = 0
                    break
            if unique == 1:
                relations.append([atom1, atom2, atom2, atom1])
                passed.append([atom1, atom2, np.real(new_eig)])
    FCM_operations = []
    for entry, r in enumerate(relations):
        FCM_operations.append(r)
        FCM_operations[entry].append([])
        good = 0
        for op in uniq_point_ops:
            new = op.transform_tensor(self.fcm[r[2]][r[3]])
            if np.allclose(new, self.fcm[r[0]][r[1]], atol=opstol):
                FCM_operations[entry][4].append(op)
                good = 1
        if r[0] == r[3] and r[1] == r[2]:
            good = 1
        if r[0] == r[2] and r[1] == r[3]:
            good = 1
        if good == 0:
            FCM_operations[entry] = [r[0], r[1], r[3], r[2]]
            FCM_operations[entry].append([])
            for op in uniq_point_ops:
                new = op.transform_tensor(self.fcm[r[2]][r[3]])
                if np.allclose(new.T, self.fcm[r[0]][r[1]], atol=opstol):
                    FCM_operations[entry][4].append(op)
    self.FCM_operations = FCM_operations
    return FCM_operations