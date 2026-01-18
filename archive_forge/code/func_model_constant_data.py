import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def model_constant_data(x, y=None):
    beta = pyro.sample('beta', dist.Normal(torch.ones(2), 3))
    pyro.sample('y', dist.Normal(x.matmul(beta), 1), obs=y)