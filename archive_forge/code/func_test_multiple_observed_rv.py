import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_multiple_observed_rv(self):
    y1 = torch.randn(10)
    y2 = torch.randn(10)

    def model_example_multiple_obs(y1=None, y2=None):
        x = pyro.sample('x', dist.Normal(1, 3))
        pyro.sample('y1', dist.Normal(x, 1), obs=y1)
        pyro.sample('y2', dist.Normal(x, 1), obs=y2)
    nuts_kernel = pyro.infer.NUTS(model_example_multiple_obs)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=10)
    mcmc.run(y1=y1, y2=y2)
    inference_data = from_pyro(mcmc)
    test_dict = {'posterior': ['x'], 'sample_stats': ['diverging'], 'log_likelihood': ['y1', 'y2'], 'observed_data': ['y1', 'y2']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails
    assert not hasattr(inference_data.sample_stats, 'log_likelihood')