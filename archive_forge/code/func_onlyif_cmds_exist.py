import os
import shutil
import sys
import tempfile
import unittest
from importlib import import_module
from decorator import decorator
from .ipunittest import ipdoctest, ipdocstring
def onlyif_cmds_exist(*commands):
    """
    Decorator to skip test when at least one of `commands` is not found.
    """
    assert os.environ.get('IPTEST_WORKING_DIR', None) is None, 'iptest deprecated since IPython 8.0'
    for cmd in commands:
        reason = f"This test runs only if command '{cmd}' is installed"
        if not shutil.which(cmd):
            import pytest
            return pytest.mark.skip(reason=reason)
    return null_deco