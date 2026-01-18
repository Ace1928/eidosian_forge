import numpy as np
def retarded(self, energy):
    return self.selfenergy_e[self.energies.searchsorted(energy)] * self.S