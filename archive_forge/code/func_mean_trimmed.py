import numbers
import numpy as np
@property
def mean_trimmed(self):
    """mean of trimmed data
        """
    return np.mean(self.data_sorted[tuple(self.sl)], self.axis)