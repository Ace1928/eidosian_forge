import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def model_example_multiple_obs(y1=None, y2=None):
    x = pyro.sample('x', dist.Normal(1, 3))
    pyro.sample('y1', dist.Normal(x, 1), obs=y1)
    pyro.sample('y2', dist.Normal(x, 1), obs=y2)