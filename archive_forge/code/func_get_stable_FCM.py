from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_stable_FCM(self, fcm, fcmasum=10):
    """
        Generate a symmetrized force constant matrix that obeys the objects symmetry
        constraints, has no unstable modes and also obeys the acoustic sum rule through an
        iterative procedure.

        Args:
            fcm (numpy array): unsymmetrized force constant matrix
            fcmasum (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            3Nx3N numpy array representing the force constant matrix
        """
    check = 0
    count = 0
    while check == 0:
        if count > 20:
            check = 1
            break
        eigs, vecs = np.linalg.eig(fcm)
        max_eig = np.max(-1 * eigs)
        eig_sort = np.argsort(np.abs(eigs))
        for idx in range(3, len(eigs)):
            if eigs[eig_sort[idx]] > 1e-06:
                eigs[eig_sort[idx]] = -1 * max_eig * np.random.rand()
        diag = np.real(np.eye(len(fcm)) * eigs)
        fcm = np.real(np.matmul(np.matmul(vecs, diag), vecs.T))
        fcm = self.get_symmetrized_FCM(fcm)
        fcm = self.get_asum_FCM(fcm)
        eigs, vecs = np.linalg.eig(fcm)
        unstable_modes = 0
        eig_sort = np.argsort(np.abs(eigs))
        for idx in range(3, len(eigs)):
            if eigs[eig_sort[idx]] > 1e-06:
                unstable_modes = 1
        if unstable_modes == 1:
            count = count + 1
            continue
        check = 1
    return fcm