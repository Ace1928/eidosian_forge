import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def estimate_mu(self, atoms, H=None):
    """Estimate optimal preconditioner coefficient \\mu

        \\mu is estimated from a numerical solution of

            [dE(p+v) -  dE(p)] \\cdot v = \\mu < P1 v, v >

        with perturbation

            v(x,y,z) = H P_lowest_nonzero_eigvec(x, y, z)

            or

            v(x,y,z) = H (sin(x / Lx), sin(y / Ly), sin(z / Lz))

        After the optimal \\mu is found, self.mu will be set to its value.

        If `atoms` is an instance of Filter an additional \\mu_c
        will be computed for the cell degrees of freedom .

        Args:
            atoms: Atoms object for initial system

            H: 3x3 array or None
                Magnitude of deformation to apply.
                Default is 1e-2*rNN*np.eye(3)

        Returns:
            mu   : float
            mu_c : float or None
        """
    logfile = self.logfile
    if self.dim != 3:
        raise ValueError('Automatic calculation of mu only possible for three-dimensional preconditioners. Try setting mu manually instead.')
    if self.r_NN is None:
        self.r_NN = estimate_nearest_neighbour_distance(atoms, self.neighbor_list)
    if H is None:
        H = 0.01 * self.r_NN * np.eye(3)
    p = atoms.get_positions()
    if self.estimate_mu_eigmode:
        self.mu = 1.0
        self.mu_c = 1.0
        c_stab = self.c_stab
        self.c_stab = 0.0
        if isinstance(atoms, Filter):
            n = len(atoms.atoms)
        else:
            n = len(atoms)
        self._make_sparse_precon(atoms, initial_assembly=True)
        self.P = self.P[:3 * n, :3 * n]
        eigvals, eigvecs = sparse.linalg.eigsh(self.P, k=4, which='SM')
        logfile.write('estimate_mu(): lowest 4 eigvals = %f %f %f %f\n' % (eigvals[0], eigvals[1], eigvals[2], eigvals[3]))
        if any(eigvals[0:3] > 1e-06):
            raise ValueError('First 3 eigenvalues of preconditioner matrixdo not correspond to translational modes.')
        elif eigvals[3] < 1e-06:
            raise ValueError('Fourth smallest eigenvalue of preconditioner matrix is too small, increase r_cut.')
        x = np.zeros(n)
        for i in range(n):
            x[i] = eigvecs[:, 3][3 * i]
        x = x / np.linalg.norm(x)
        if x[0] < 0:
            x = -x
        v = np.zeros(3 * len(atoms))
        for i in range(n):
            v[3 * i] = x[i]
            v[3 * i + 1] = x[i]
            v[3 * i + 2] = x[i]
        v = v / np.linalg.norm(v)
        v = v.reshape((-1, 3))
        self.c_stab = c_stab
    else:
        Lx, Ly, Lz = [p[:, i].max() - p[:, i].min() for i in range(3)]
        logfile.write('estimate_mu(): Lx=%.1f Ly=%.1f Lz=%.1f\n' % (Lx, Ly, Lz))
        x, y, z = p.T
        sine_vr = [x, y, z]
        for i, L in enumerate([Lx, Ly, Lz]):
            if L == 0:
                warnings.warn('Cell length L[%d] == 0. Setting H[%d,%d] = 0.' % (i, i, i))
                H[i, i] = 0.0
            else:
                sine_vr[i] = np.sin(sine_vr[i] / L)
        v = np.dot(H, sine_vr).T
    natoms = len(atoms)
    if isinstance(atoms, Filter):
        natoms = len(atoms.atoms)
        eps = H / self.r_NN
        v[natoms:, :] = eps
    v1 = v.reshape(-1)
    dE_p = -atoms.get_forces().reshape(-1)
    atoms_v = atoms.copy()
    atoms_v.calc = atoms.calc
    if isinstance(atoms, Filter):
        atoms_v = atoms.__class__(atoms_v)
        if hasattr(atoms, 'constant_volume'):
            atoms_v.constant_volume = atoms.constant_volume
    atoms_v.set_positions(p + v)
    dE_p_plus_v = -atoms_v.get_forces().reshape(-1)
    LHS = (dE_p_plus_v - dE_p) * v1
    self.mu = 1.0
    self.mu_c = 1.0
    self._make_sparse_precon(atoms, initial_assembly=True)
    RHS = self.P.dot(v1) * v1
    self.mu = longsum(LHS[:3 * natoms]) / longsum(RHS[:3 * natoms])
    if self.mu < 1.0:
        logfile.write('estimate_mu(): mu (%.3f) < 1.0, capping at mu=1.0' % self.mu)
        self.mu = 1.0
    if isinstance(atoms, Filter):
        self.mu_c = longsum(LHS[3 * natoms:]) / longsum(RHS[3 * natoms:])
        if self.mu_c < 1.0:
            logfile.write('estimate_mu(): mu_c (%.3f) < 1.0, capping at mu_c=1.0\n' % self.mu_c)
            self.mu_c = 1.0
    logfile.write('estimate_mu(): mu=%r, mu_c=%r\n' % (self.mu, self.mu_c))
    self.P = None
    return (self.mu, self.mu_c)