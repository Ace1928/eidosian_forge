import numbers
import numpy as np
def reset_fraction(self, frac):
    """create a TrimmedMean instance with a new trimming fraction

        This reuses the sorted array from the current instance.
        """
    tm = TrimmedMean(self.data_sorted, frac, is_sorted=True, axis=self.axis)
    tm.data = self.data
    return tm