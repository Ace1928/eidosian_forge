import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_input_types5(self, paths, observed_data_paths):
    """Check input types (change, see earlier)

        posterior_predictive is None
        prior_predictive is None
        """
    for key, path in paths.items():
        if 'eight' not in key:
            continue
        inference_data = self.get_inference_data(posterior=path, posterior_predictive=None, prior=path, prior_predictive=None, observed_data=observed_data_paths[0], observed_data_var=['y'], log_likelihood=['y_hat'], coords={'school': np.arange(8), 'log_lik_dim': np.arange(8)}, dims={'theta': ['school'], 'y': ['school'], 'log_lik': ['log_lik_dim'], 'y_hat': ['school'], 'eta': ['school']})
        test_dict = {'posterior': ['mu', 'tau', 'theta_tilde', 'theta', 'log_lik'], 'prior': ['mu', 'tau', 'theta_tilde', 'theta'], 'log_likelihood': ['y_hat', '~log_lik'], 'observed_data': ['y'], 'sample_stats_prior': ['lp']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails