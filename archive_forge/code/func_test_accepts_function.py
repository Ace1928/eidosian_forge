import inspect
import unittest
from traits.api import (
def test_accepts_function(self):
    MyBaseCallable(value=lambda x: x)