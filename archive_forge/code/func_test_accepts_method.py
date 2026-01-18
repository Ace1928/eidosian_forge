import inspect
import unittest
from traits.api import (
def test_accepts_method(self):
    MyBaseCallable(value=Dummy.instance_method)