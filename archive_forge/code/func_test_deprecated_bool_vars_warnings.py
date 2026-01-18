import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
@pytest.mark.parametrize('deprecated_var, new_var', [(cfg.ExperimentalGroupbyImpl, cfg.RangePartitioning), (cfg.ExperimentalNumPyAPI, cfg.ModinNumpy), (cfg.RangePartitioningGroupby, cfg.RangePartitioning)])
def test_deprecated_bool_vars_warnings(deprecated_var, new_var):
    """Test that deprecated parameters are raising `FutureWarnings` and their replacements don't."""
    old_depr_val = deprecated_var.get()
    old_new_var = new_var.get()
    try:
        reset_vars(deprecated_var, new_var)
        with pytest.warns(FutureWarning):
            deprecated_var.get()
        with pytest.warns(FutureWarning):
            deprecated_var.put(False)
        with unittest.mock.patch.dict(os.environ, {deprecated_var.varname: '1'}):
            with pytest.warns(FutureWarning):
                _check_vars()
        reset_vars(deprecated_var, new_var)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            new_var.get()
            new_var.put(False)
            with unittest.mock.patch.dict(os.environ, {new_var.varname: '1'}):
                _check_vars()
    finally:
        deprecated_var.put(old_depr_val)
        new_var.put(old_new_var)