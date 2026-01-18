import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_get_function_defaults():

    def f(x, y=3, b=4):
        pass
    assert six.get_function_defaults(f) == (3, 4)