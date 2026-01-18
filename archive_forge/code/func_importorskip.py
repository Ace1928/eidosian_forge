import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def importorskip(modname: str, minversion: Optional[str]=None, reason: Optional[str]=None) -> Any:
    """Import and return the requested module ``modname``.

        Doesn't allow skips on CI machine.
        Borrowed and modified from ``pytest.importorskip``.
    :param str modname: the name of the module to import
    :param str minversion: if given, the imported module's ``__version__``
        attribute must be at least this minimal version, otherwise the test is
        still skipped.
    :param str reason: if given, this reason is shown as the message when the
        module cannot be imported.
    :returns: The imported module. This should be assigned to its canonical
        name.
    Example::
        docutils = pytest.importorskip("docutils")
    """
    ARVIZ_CI_MACHINE = running_on_ci()
    if not ARVIZ_CI_MACHINE:
        return pytest.importorskip(modname=modname, minversion=minversion, reason=reason)
    import warnings
    compile(modname, '', 'eval')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        __import__(modname)
    mod = sys.modules[modname]
    if minversion is None:
        return mod
    verattr = getattr(mod, '__version__', None)
    if verattr is None or Version(verattr) < Version(minversion):
        raise Skipped('module %r has __version__ %r, required is: %r' % (modname, verattr, minversion), allow_module_level=True)
    return mod