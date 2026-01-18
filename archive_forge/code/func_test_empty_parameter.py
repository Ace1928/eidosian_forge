import importlib
from collections import OrderedDict
import numpy as np
import pytest
from ... import from_pystan
from ...data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from ..helpers import (  # pylint: disable=unused-import
def test_empty_parameter(self):
    model_code = '\n            parameters {\n                real y;\n                vector[3] x;\n                vector[0] a;\n                vector[2] z;\n            }\n            model {\n                y ~ normal(0,1);\n            }\n        '
    if pystan_version() == 2:
        from pystan import StanModel
        model = StanModel(model_code=model_code)
        fit = model.sampling(iter=500, chains=2, check_hmc_diagnostics=False)
    else:
        import stan
        model = stan.build(model_code)
        fit = model.sample(num_samples=500, num_chains=2)
    posterior = from_pystan(posterior=fit)
    test_dict = {'posterior': ['y', 'x', 'z', '~a'], 'sample_stats': ['diverging']}
    fails = check_multiple_attrs(test_dict, posterior)
    assert not fails