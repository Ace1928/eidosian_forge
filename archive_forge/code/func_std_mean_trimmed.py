import numbers
import numpy as np
@property
def std_mean_trimmed(self):
    """standard error of trimmed mean
        """
    se = np.sqrt(self.var_winsorized / self.nobs_reduced)
    se *= np.sqrt(self.nobs / self.nobs_reduced)
    return se