from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_dict_methods(self):
    mock = Mock()
    self.assertRaises(TypeError, lambda: mock['foo'])

    def _del():
        del mock['foo']

    def _set():
        mock['foo'] = 3
    self.assertRaises(TypeError, _del)
    self.assertRaises(TypeError, _set)
    _dict = {}

    def getitem(s, name):
        return _dict[name]

    def setitem(s, name, value):
        _dict[name] = value

    def delitem(s, name):
        del _dict[name]
    mock.__setitem__ = setitem
    mock.__getitem__ = getitem
    mock.__delitem__ = delitem
    self.assertRaises(KeyError, lambda: mock['foo'])
    mock['foo'] = 'bar'
    self.assertEqual(_dict, {'foo': 'bar'})
    self.assertEqual(mock['foo'], 'bar')
    del mock['foo']
    self.assertEqual(_dict, {})