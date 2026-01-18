import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_mock_add_spec(self):

    class _One(object):
        one = 1

    class _Two(object):
        two = 2

    class Anything(object):
        one = two = three = 'four'
    klasses = [Mock, MagicMock, NonCallableMock, NonCallableMagicMock]
    for Klass in list(klasses):
        klasses.append(lambda K=Klass: K(spec=Anything))
        klasses.append(lambda K=Klass: K(spec_set=Anything))
    for Klass in klasses:
        for kwargs in (dict(), dict(spec_set=True)):
            mock = Klass()
            (mock.one, mock.two, mock.three)
            for One, Two in [(_One, _Two), (['one'], ['two'])]:
                for kwargs in (dict(), dict(spec_set=True)):
                    mock.mock_add_spec(One, **kwargs)
                    mock.one
                    self.assertRaises(AttributeError, getattr, mock, 'two')
                    self.assertRaises(AttributeError, getattr, mock, 'three')
                    if 'spec_set' in kwargs:
                        self.assertRaises(AttributeError, setattr, mock, 'three', None)
                    mock.mock_add_spec(Two, **kwargs)
                    self.assertRaises(AttributeError, getattr, mock, 'one')
                    mock.two
                    self.assertRaises(AttributeError, getattr, mock, 'three')
                    if 'spec_set' in kwargs:
                        self.assertRaises(AttributeError, setattr, mock, 'three', None)