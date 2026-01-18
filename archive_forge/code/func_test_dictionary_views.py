import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_dictionary_views():
    d = dict(zip(range(10), range(11, 20)))
    for name in ('keys', 'values', 'items'):
        meth = getattr(six, 'view' + name)
        view = meth(d)
        assert set(view) == set(getattr(d, name)())