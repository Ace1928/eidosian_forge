import operator
import sys
import types
import unittest
import abc
import pytest
import six
def stock_method_name(iterwhat):
    """Given a method suffix like "lists" or "values", return the name
        of the dict method that delivers those on the version of Python
        we're running in."""
    if six.PY3:
        return iterwhat
    return 'iter' + iterwhat