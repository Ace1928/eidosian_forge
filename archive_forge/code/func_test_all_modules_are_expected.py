import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def test_all_modules_are_expected():
    """
    Test that we don't add anything that looks like a new public module by
    accident.  Check is based on filenames.
    """
    modnames = []
    for _, modname, ispkg in pkgutil.walk_packages(path=scipy.__path__, prefix=scipy.__name__ + '.', onerror=None):
        if is_unexpected(modname) and modname not in SKIP_LIST:
            modnames.append(modname)
    if modnames:
        raise AssertionError(f'Found unexpected modules: {modnames}')