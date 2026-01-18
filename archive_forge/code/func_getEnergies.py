import numpy as np
def getEnergies(self):
    """
        "energies" are just normalized eigenvectors
        """
    v = self.getEigenvalues()
    return v / np.sum(v)