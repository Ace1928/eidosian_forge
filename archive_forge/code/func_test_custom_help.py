import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
def test_custom_help(make_custom_envvar):
    assert 'MODIN_CUSTOM' in make_custom_envvar.get_help()
    assert 'custom var' in make_custom_envvar.get_help()