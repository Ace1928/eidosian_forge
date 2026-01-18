from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
@contextmanager
def cleanup_user_ns(**kwargs):
    """
    On exit delete all the keys that were not in user_ns before entering.

    It does not restore old values !

    Parameters
    ----------

    **kwargs
        used to update ip.user_ns

    """
    try:
        known = set(ip.user_ns.keys())
        ip.user_ns.update(kwargs)
        yield
    finally:
        added = set(ip.user_ns.keys()) - known
        for k in added:
            del ip.user_ns[k]