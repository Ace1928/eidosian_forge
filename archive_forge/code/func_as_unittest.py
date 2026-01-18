import os
import shutil
import sys
import tempfile
import unittest
from importlib import import_module
from decorator import decorator
from .ipunittest import ipdoctest, ipdocstring
def as_unittest(func):
    """Decorator to make a simple function into a normal test via unittest."""

    class Tester(unittest.TestCase):

        def test(self):
            func()
    Tester.__name__ = func.__name__
    return Tester