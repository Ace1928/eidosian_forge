from David Huard's scipy sandbox, also attached to a ticket and
import scipy.interpolate as interpolate
import numpy as np
def empiricalcdf(self, data=None, method='Hazen'):
    """Return the empirical cdf.

        Methods available:
            Hazen:       (i-0.5)/N
                Weibull:     i/(N+1)
            Chegodayev:  (i-.3)/(N+.4)
            Cunnane:     (i-.4)/(N+.2)
            Gringorten:  (i-.44)/(N+.12)
            California:  (i-1)/N

        Where i goes from 1 to N.
        """
    if data is None:
        data = self.data
        i = self.ranking
    else:
        i = np.argsort(np.argsort(data)) + 1.0
    N = len(data)
    method = method.lower()
    if method == 'hazen':
        cdf = (i - 0.5) / N
    elif method == 'weibull':
        cdf = i / (N + 1.0)
    elif method == 'california':
        cdf = (i - 1.0) / N
    elif method == 'chegodayev':
        cdf = (i - 0.3) / (N + 0.4)
    elif method == 'cunnane':
        cdf = (i - 0.4) / (N + 0.2)
    elif method == 'gringorten':
        cdf = (i - 0.44) / (N + 0.12)
    else:
        raise ValueError('Unknown method. Choose among Weibull, Hazen,Chegodayev, Cunnane, Gringorten and California.')
    return cdf