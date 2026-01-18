import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
@pytest.mark.parametrize('deprecated_var, new_var', [(cfg.ExperimentalGroupbyImpl, cfg.RangePartitioningGroupby), (cfg.ExperimentalNumPyAPI, cfg.ModinNumpy)])
@pytest.mark.parametrize('get_depr_first', [True, False])
def test_deprecated_bool_vars_equals(deprecated_var, new_var, get_depr_first):
    """
    Test that deprecated parameters always have values equal to the new replacement parameters.

    Parameters
    ----------
    deprecated_var : EnvironmentVariable
    new_var : EnvironmentVariable
    get_depr_first : bool
        Defines an order in which the ``.get()`` method should be called when comparing values.
        If ``True``: get deprecated value at first ``deprecated_var.get() == new_var.get() == value``.
        If ``False``: get new value at first ``new_var.get() == deprecated_var.get() == value``.
        The logic of the ``.get()`` method depends on which parameter was initialized first,
        that's why it's worth testing both cases.
    """
    old_depr_val = deprecated_var.get()
    old_new_var = new_var.get()

    def get_values():
        return (deprecated_var.get(), new_var.get()) if get_depr_first else (new_var.get(), deprecated_var.get())
    try:
        reset_vars(deprecated_var, new_var)
        deprecated_var.put(True)
        val1, val2 = get_values()
        assert val1 == val2 == True
        new_var.put(False)
        val1, val2 = get_values()
        assert val1 == val2 == False
        new_var.put(True)
        val1, val2 = get_values()
        assert val1 == val2 == True
        deprecated_var.put(False)
        val1, val2 = get_values()
        assert val1 == val2 == False
        reset_vars(deprecated_var, new_var)
        new_var.put(True)
        val1, val2 = get_values()
        assert val1 == val2 == True
        deprecated_var.put(False)
        val1, val2 = get_values()
        assert val1 == val2 == False
        deprecated_var.put(True)
        val1, val2 = get_values()
        assert val1 == val2 == True
        new_var.put(False)
        val1, val2 = get_values()
        assert val1 == val2 == False
        reset_vars(deprecated_var, new_var)
        with unittest.mock.patch.dict(os.environ, {deprecated_var.varname: 'True'}):
            val1, val2 = get_values()
            assert val1 == val2 == True
            new_var.put(False)
            val1, val2 = get_values()
            assert val1 == val2 == False
            new_var.put(True)
            val1, val2 = get_values()
            assert val1 == val2 == True
            deprecated_var.put(False)
            val1, val2 = get_values()
            assert val1 == val2 == False
        reset_vars(deprecated_var, new_var)
        with unittest.mock.patch.dict(os.environ, {new_var.varname: 'True'}):
            val1, val2 = get_values()
            assert val1 == val2 == True
            deprecated_var.put(False)
            val1, val2 = get_values()
            assert val1 == val2 == False
            deprecated_var.put(True)
            val1, val2 = get_values()
            assert val1 == val2 == True
            new_var.put(False)
            val1, val2 = get_values()
            assert val1 == val2 == False
    finally:
        deprecated_var.put(old_depr_val)
        new_var.put(old_new_var)