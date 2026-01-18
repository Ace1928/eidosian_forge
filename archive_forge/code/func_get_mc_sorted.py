from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
def get_mc_sorted(self):
    if not hasattr(self, 'mcressort'):
        self.mcressort = np.sort(self.mcres, axis=0)
    return self.mcressort