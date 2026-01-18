import numpy as np
import numpy.linalg as la
def dK_dl_matrix(self, x1, x2):
    k = np.asarray(self.dK_dl_k(x1, x2)).reshape((1, 1))
    j2 = self.dK_dl_j(x1, x2).reshape(1, -1)
    j1 = self.dK_dl_j(x2, x1).reshape(-1, 1)
    h = self.dK_dl_h(x1, x2)
    return np.block([[k, j2], [j1, h]]) * self.kernel_function(x1, x2)