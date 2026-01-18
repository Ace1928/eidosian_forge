import os
import shutil
import sys
import tempfile
import unittest
from importlib import import_module
from decorator import decorator
from .ipunittest import ipdoctest, ipdocstring
def skipif(skip_condition, msg=None):
    """Make function raise SkipTest exception if skip_condition is true

    Parameters
    ----------

    skip_condition : bool or callable
      Flag to determine whether to skip test. If the condition is a
      callable, it is used at runtime to dynamically make the decision. This
      is useful for tests that may require costly imports, to delay the cost
      until the test suite is actually executed.
    msg : string
      Message to give on raising a SkipTest exception.

    Returns
    -------
    decorator : function
      Decorator, which, when applied to a function, causes SkipTest
      to be raised when the skip_condition was True, and the function
      to be called normally otherwise.
    """
    if msg is None:
        msg = 'Test skipped due to test condition.'
    import pytest
    assert isinstance(skip_condition, bool)
    return pytest.mark.skipif(skip_condition, reason=msg)