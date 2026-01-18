import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_inference_data_no_posterior(self, data, eight_schools_params, predictions_data, predictions_params):
    posterior_samples = data.obj.get_samples()
    model = data.obj.kernel.model
    posterior_predictive = Predictive(model, posterior_samples)(eight_schools_params['J'], torch.from_numpy(eight_schools_params['sigma']).float())
    prior = Predictive(model, num_samples=500)(eight_schools_params['J'], torch.from_numpy(eight_schools_params['sigma']).float())
    predictions = predictions_data
    constant_data = {'J': 8, 'sigma': eight_schools_params['sigma']}
    predictions_constant_data = predictions_params
    inference_data = from_pyro(prior=prior)
    test_dict = {'prior': ['mu', 'tau', 'eta']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails, f'only prior: {fails}'
    inference_data = from_pyro(posterior_predictive=posterior_predictive)
    test_dict = {'posterior_predictive': ['obs']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails, f'only posterior_predictive: {fails}'
    inference_data = from_pyro(predictions=predictions)
    test_dict = {'predictions': ['obs']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails, f'only predictions: {fails}'
    inference_data = from_pyro(constant_data=constant_data)
    test_dict = {'constant_data': ['J', 'sigma']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails, f'only constant_data: {fails}'
    inference_data = from_pyro(predictions_constant_data=predictions_constant_data)
    test_dict = {'predictions_constant_data': ['J', 'sigma']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails, f'only predictions_constant_data: {fails}'
    idata = from_pyro(prior=prior, posterior_predictive=posterior_predictive, coords={'school': np.arange(eight_schools_params['J'])}, dims={'theta': ['school'], 'eta': ['school']})
    test_dict = {'posterior_predictive': ['obs'], 'prior': ['mu', 'tau', 'eta', 'obs']}
    fails = check_multiple_attrs(test_dict, idata)
    assert not fails, f'prior and posterior_predictive: {fails}'