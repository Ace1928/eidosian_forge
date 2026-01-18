import numpy as np
def subtractPC(self, pc, vals=None):
    """
        pc can be a scalar or any sequence of pc indecies

        if vals is None, the source data is self.A, else whatever is in vals
        (which must be p x m)
        """
    if vals is None:
        vals = self.A
    else:
        vals = vals.T
        if vals.shape[1] != self.A.shape[1]:
            raise ValueError('vals do not have the correct number of components')
    pcs = self.project()
    zpcs = np.zeros_like(pcs)
    zpcs[:, pc] = pcs[:, pc]
    upc = self.deproject(zpcs, False)
    A = vals.T - upc
    B = A.T * np.std(self.M, axis=0)
    return B + np.mean(self.A, axis=0)