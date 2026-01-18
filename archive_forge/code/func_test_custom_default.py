import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
def test_custom_default(make_custom_envvar):
    assert make_custom_envvar.get() == 10