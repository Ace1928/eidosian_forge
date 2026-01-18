import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
@contextlib.contextmanager
def assertDeprecated(self):
    """
        Assert that the code inside the with block is deprecated.  Intended
        for testing uses of traits.util.deprecated.deprecated.

        """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        yield w
    self.assertGreater(len(w), 0, msg='Expected a DeprecationWarning, but none was issued')