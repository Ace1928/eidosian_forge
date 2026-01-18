import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@property
def asserts_when_called(self):
    raise AssertionError('getter method called')