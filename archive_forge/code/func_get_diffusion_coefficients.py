import numpy as np
def get_diffusion_coefficients(self):
    """
        
        Returns diffusion coefficients for atoms (in alphabetical order) along with standard deviation.
        
        All data is currently passed out in units of Å^2/<ASE time units>
        To convert into Å^2/fs => multiply by ase.units.fs
        To convert from Å^2/fs to cm^2/s => multiply by (10^-8)^2 / 10^-15 = 10^-1
        
        """
    slopes = [np.mean(self.slopes[sym_index]) for sym_index in range(self.no_of_types_of_atoms)]
    std = [np.std(self.slopes[sym_index]) for sym_index in range(self.no_of_types_of_atoms)]
    return (slopes, std)