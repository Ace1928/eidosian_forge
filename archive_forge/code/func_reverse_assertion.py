import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
@contextlib.contextmanager
def reverse_assertion(context, msg):
    context.__enter__()
    try:
        yield context
    finally:
        try:
            context.__exit__(None, None, None)
        except AssertionError:
            pass
        else:
            raise context.failureException(msg)