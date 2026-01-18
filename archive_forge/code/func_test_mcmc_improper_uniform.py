from collections import namedtuple
import numpy as np
import pytest
from ...data.io_numpyro import from_numpyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_mcmc_improper_uniform(self):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        x = numpyro.sample('x', dist.ImproperUniform(dist.constraints.positive, (), ()))
        return numpyro.sample('y', dist.Normal(x, 1), obs=1.0)
    mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
    mcmc.run(PRNGKey(0))
    inference_data = from_numpyro(mcmc)
    assert inference_data.observed_data