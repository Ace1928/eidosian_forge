import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
@pytest.fixture
def set_custom_envvar(make_custom_envvar):
    os.environ[make_custom_envvar.varname] = '  custom  '
    yield ('Custom' if make_custom_envvar.type is str else '  custom  ')
    del os.environ[make_custom_envvar.varname]