import numpy as np
import pandas as pd
from scipy import stats
def summary_frame(self, alpha=0.05):
    """
        Summary frame of mean, variance and confidence interval.

        Returns
        -------
        DataFrame
            DataFrame containing four columns:

            * mean
            * mean_se
            * mean_ci_lower
            * mean_ci_upper

        Notes
        -----
        Fixes alpha to 0.05 so that the confidence interval should have 95%
        coverage.
        """
    ci_mean = np.asarray(self.conf_int(alpha=alpha))
    lower, upper = (ci_mean[:, 0], ci_mean[:, 1])
    to_include = {'mean': self.predicted_mean, 'mean_se': self.se_mean, 'mean_ci_lower': lower, 'mean_ci_upper': upper}
    return pd.DataFrame(to_include)