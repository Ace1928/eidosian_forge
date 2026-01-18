import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def cases_test_discrete_integer_shapes():
    integrality_exceptions = {'nbinom': {'n'}, 'betanbinom': {'n'}}
    seen = set()
    for distname, shapes in distdiscrete:
        if distname in seen:
            continue
        seen.add(distname)
        try:
            dist = getattr(stats, distname)
        except TypeError:
            continue
        shape_info = dist._shape_info()
        for i, shape in enumerate(shape_info):
            if shape.name in integrality_exceptions.get(distname, set()) or not shape.integrality:
                continue
            yield (distname, shape.name, shapes)