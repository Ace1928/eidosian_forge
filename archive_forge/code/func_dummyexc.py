from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def dummyexc():
    raise SomeError()