import numpy as np
from ase.io.jsonio import read_json, write_json
def get_averaged_current(self, bias, z):
    """Calculate avarage current at height z (in Angstrom).

        Use this to get an idea of what current to use when scanning."""
    self.calculate_ldos(bias)
    nz = self.ldos.shape[2]
    n = z / self.cell[2, 2] * nz
    dn = n - np.floor(n)
    n = int(n) % nz
    return (1 - dn) * self.ldos[:, :, n].mean() + dn * self.ldos[:, :, (n + 1) % nz].mean()