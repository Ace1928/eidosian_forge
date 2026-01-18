from collections import namedtuple
import numpy as np
import pytest
from ...data.io_numpyro import from_numpyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
@pytest.mark.parametrize('nchains', [1, 2])
@pytest.mark.parametrize('thin', [1, 2, 3, 5, 10])
def test_mcmc_with_thinning(self, nchains, thin):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    x = np.random.normal(10, 3, size=100)

    def model(x):
        numpyro.sample('x', dist.Normal(numpyro.sample('loc', dist.Uniform(0, 20)), numpyro.sample('scale', dist.Uniform(0, 20))), obs=x)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=400, num_chains=nchains, thinning=thin)
    mcmc.run(PRNGKey(0), x=x)
    inference_data = from_numpyro(mcmc)
    assert inference_data.posterior['loc'].shape == (nchains, 400 // thin)