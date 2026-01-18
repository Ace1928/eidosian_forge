import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
def reset_vars(*vars: tuple[cfg.Parameter]):
    """
    Reset value for the passed parameters.

    Parameters
    ----------
    *vars : tuple[Parameter]
    """
    for var in vars:
        var._value = _UNSET
        _ = os.environ.pop(var.varname, None)