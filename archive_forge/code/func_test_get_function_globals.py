import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_get_function_globals():

    def f():
        pass
    assert six.get_function_globals(f) is globals()