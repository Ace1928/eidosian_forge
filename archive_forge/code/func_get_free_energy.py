import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
def get_free_energy(self, T, method='classical'):
    """Get analytic vibrational free energy for the spring system.

        Parameters
        ----------
        T : float
            temperature (K)
        method : str
            method for free energy computation; 'classical' or 'QM'.
        """
    F = 0.0
    masses, counts = np.unique(self.atoms.get_masses(), return_counts=True)
    for m, c in zip(masses, counts):
        F += c * SpringCalculator.compute_Einstein_solid_free_energy(self.k, m, T, method)
    return F