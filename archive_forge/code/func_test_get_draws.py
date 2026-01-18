import importlib
from collections import OrderedDict
import numpy as np
import pytest
from ... import from_pystan
from ...data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from ..helpers import (  # pylint: disable=unused-import
def test_get_draws(self, data):
    fit = data.obj
    if pystan_version() == 2:
        draws, _ = get_draws(fit, variables=['theta', 'theta'])
    else:
        draws, _ = get_draws_stan3(fit, variables=['theta', 'theta'])
    assert draws.get('theta') is not None