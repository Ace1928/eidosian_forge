import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_exec_():

    def f():
        l = []
        six.exec_('l.append(1)')
        assert l == [1]
    f()
    ns = {}
    six.exec_('x = 42', ns)
    assert ns['x'] == 42
    glob = {}
    loc = {}
    six.exec_('global y; y = 42; x = 12', glob, loc)
    assert glob['y'] == 42
    assert 'x' not in glob
    assert loc['x'] == 12
    assert 'y' not in loc