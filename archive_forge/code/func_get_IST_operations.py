from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_IST_operations(self, opstol=0.001):
    """
        Returns the symmetry operations which maps the tensors
        belonging to equivalent sites onto each other in the form
        [site index 1, site index 2, [Symmops mapping from site
        index 1 to site index 2]].

        Args:
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
    IST_operations = []
    for atom in range(len(self.ist)):
        IST_operations.append([])
        for j in range(atom):
            for op in uniq_point_ops:
                new = op.transform_tensor(self.ist[j])
                if np.allclose(new, self.ist[atom], atol=opstol):
                    IST_operations[atom].append([j, op])
    self.IST_operations = IST_operations