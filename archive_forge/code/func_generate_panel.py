import numpy as np
from . import correlation_structures as cs
def generate_panel(self):
    """
        generate endog for a random panel dataset with within correlation

        """
    random = self.random_state
    if self.y_true is None:
        self.get_y_true()
    nobs_i = self.nobs_i
    n_groups = self.n_groups
    use_balanced = True
    if use_balanced:
        noise = self.random_state.multivariate_normal(np.zeros(nobs_i), self.cov, size=n_groups).ravel()
        noise += np.repeat(self.group_means, nobs_i)
    else:
        noise = np.empty(self.nobs, np.float64)
        noise.fill(np.nan)
        for ii in range(self.n_groups):
            idx, idxupp = self.group_indices[ii:ii + 2]
            mean_i = self.group_means[ii]
            noise[idx:idxupp] = self.random_state.multivariate_normal(mean_i * np.ones(self.nobs_i), self.cov)
    endog = self.y_true + noise
    return endog