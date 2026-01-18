import numpy as np
from ase.io.jsonio import read_json, write_json
def scan2(self, bias, z, repeat=(1, 1)):
    """Constant height 2-d scan.

        Returns three 2-d arrays (x, y, I) containing x-coordinates,
        y-coordinates and currents.  These three arrays can be passed to
        matplotlibs contourf() function like this:

        >>> import matplotlib.pyplot as plt
        >>> plt.contourf(x, y, I)
        >>> plt.show()

        """
    self.calculate_ldos(bias)
    nz = self.ldos.shape[2]
    ldos = self.ldos.reshape((-1, nz))
    I = np.empty(ldos.shape[0])
    zp = z / self.cell[2, 2] * nz
    zp = int(zp) % nz
    for i, a in enumerate(ldos):
        I[i] = self.find_current(a, zp)
    s0 = I.shape = self.ldos.shape[:2]
    I = np.tile(I, repeat)
    s = I.shape
    ij = np.indices(s, dtype=float).reshape((2, -1)).T
    x, y = np.dot(ij / s0, self.cell[:2, :2]).T.reshape((2,) + s)
    return (x, y, I)