import numpy as np
def dos(self, energy):
    """Total density of states -1/pi Im(Tr(GS))"""
    if self.S is None:
        return -self.retarded(energy).imag.trace() / np.pi
    else:
        GS = self.apply_retarded(energy, self.S)
        return -GS.imag.trace() / np.pi