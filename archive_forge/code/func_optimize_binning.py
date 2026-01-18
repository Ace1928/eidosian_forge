from David Huard's scipy sandbox, also attached to a ticket and
import scipy.interpolate as interpolate
import numpy as np
def optimize_binning(self, method='Freedman'):
    """Find the optimal number of bins and update the bin countaccordingly.
        Available methods : Freedman
                            Scott
        """
    nobs = len(self.data)
    if method == 'Freedman':
        IQR = self.ppf_emp(0.75) - self.ppf_emp(0.25)
        width = 2 * IQR * nobs ** (-1.0 / 3)
    elif method == 'Scott':
        width = 3.49 * np.std(self.data) * nobs ** (-1.0 / 3)
    self.nbin = np.ptp(self.binlimit) / width
    return self.nbin