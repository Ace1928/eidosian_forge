import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
def test_ln_funcs_for_infinity(self):
    ary = np.ones(10)
    ary[1] = -1
    assert np.isinf(emcee_lnprior(ary))
    assert np.isinf(emcee_lnprob(ary, ary[2:], ary[2:])[0])