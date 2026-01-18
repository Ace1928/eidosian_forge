import inspect
import unittest
from traits.api import (
def test_accepts_none(self):
    MyBaseCallable(value=None)