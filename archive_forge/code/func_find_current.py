import numpy as np
from ase.io.jsonio import read_json, write_json
def find_current(self, ldos, z):
    """ Finds current for given LDOS at height z."""
    nz = self.ldos.shape[2]
    zp = z / self.cell[2, 2] * nz
    dz = zp - np.floor(zp)
    zp = int(zp) % nz
    ldosz = (1 - dz) * ldos[zp] + dz * ldos[(zp + 1) % nz]
    return dos2current(self.bias, ldosz)