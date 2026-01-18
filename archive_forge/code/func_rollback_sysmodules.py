import subprocess
import sys
from collections import namedtuple
from io import StringIO
from subprocess import PIPE
from typing import Any, Callable, Dict, Generator, Optional, Tuple
import pytest
from sphinx.testing import util
from sphinx.testing.util import SphinxTestApp, SphinxTestAppWrapperForSkipBuilding
@pytest.fixture
def rollback_sysmodules():
    """
    Rollback sys.modules to its value before testing to unload modules
    during tests.

    For example, used in test_ext_autosummary.py to permit unloading the
    target module to clear its cache.
    """
    try:
        sysmodules = list(sys.modules)
        yield
    finally:
        for modname in list(sys.modules):
            if modname not in sysmodules:
                sys.modules.pop(modname)