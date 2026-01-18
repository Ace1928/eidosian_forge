import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
def test_hdk_envvar():
    try:
        import pyhdk
        defaults = cfg.HdkLaunchParameters.get()
        assert defaults['enable_union'] == 1
        if version.parse(pyhdk.__version__) >= version.parse('0.6.1'):
            assert defaults['log_dir'] == 'pyhdk_log'
        del cfg.HdkLaunchParameters._value
    except ImportError:
        pass
    os.environ[cfg.HdkLaunchParameters.varname] = 'enable_union=2,enable_thrift_logs=3'
    params = cfg.HdkLaunchParameters.get()
    assert params['enable_union'] == 2
    assert params['enable_thrift_logs'] == 3
    os.environ[cfg.HdkLaunchParameters.varname] = 'unsupported=X'
    del cfg.HdkLaunchParameters._value
    params = cfg.HdkLaunchParameters.get()
    assert params['unsupported'] == 'X'
    try:
        import pyhdk
        pyhdk.buildConfig(**cfg.HdkLaunchParameters.get())
    except RuntimeError as e:
        assert str(e) == "unrecognised option '--unsupported'"
    except ImportError:
        pass
    os.environ[cfg.HdkLaunchParameters.varname] = 'enable_union=4,enable_thrift_logs=5,enable_lazy_dict_materialization=6'
    del cfg.HdkLaunchParameters._value
    params = cfg.HdkLaunchParameters.get()
    assert params['enable_union'] == 4
    assert params['enable_thrift_logs'] == 5
    assert params['enable_lazy_dict_materialization'] == 6