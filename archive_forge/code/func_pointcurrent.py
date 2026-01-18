import numpy as np
from ase.io.jsonio import read_json, write_json
def pointcurrent(self, bias, x, y, z):
    """Current for a single x, y, z position for a given bias."""
    self.calculate_ldos(bias)
    nx = self.ldos.shape[0]
    ny = self.ldos.shape[1]
    nz = self.ldos.shape[2]
    xp = x / np.linalg.norm(self.cell[0]) * nx
    dx = xp - np.floor(xp)
    xp = int(xp) % nx
    yp = y / np.linalg.norm(self.cell[1]) * ny
    dy = yp - np.floor(yp)
    yp = int(yp) % ny
    zp = z / np.linalg.norm(self.cell[2]) * nz
    dz = zp - np.floor(zp)
    zp = int(zp) % nz
    xyzldos = (1 - dx + (1 - dy) + (1 - dz)) * self.ldos[xp, yp, zp] + dx * self.ldos[(xp + 1) % nx, yp, zp] + dy * self.ldos[xp, (yp + 1) % ny, zp] + dz * self.ldos[xp, yp, (zp + 1) % nz]
    return dos2current(bias, xyzldos)