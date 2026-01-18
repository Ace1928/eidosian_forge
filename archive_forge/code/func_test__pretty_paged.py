import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__pretty_paged(self):
    self.cmd_base.limit = '5'
    func = mock.Mock(return_value=None)
    self.cmd_base.verbose = True
    self.assertIsNone(self.cmd_base._pretty_paged(func))
    self.cmd_base.verbose = False

    class MockIterable(collections.abc.Iterable):
        links = ['item']
        count = 1

        def __iter__(self):
            return ['item1']

        def __len__(self):
            return self.count
    ret = MockIterable()
    func = mock.Mock(return_value=ret)
    self.assertIsNone(self.cmd_base._pretty_paged(func))
    ret.count = 0
    self.assertIsNone(self.cmd_base._pretty_paged(func))
    func = mock.Mock(side_effect=ValueError)
    self.assertIsNone(self.cmd_base._pretty_paged(func))
    self.cmd_base.debug = True
    self.cmd_base.marker = mock.Mock()
    self.assertRaises(ValueError, self.cmd_base._pretty_paged, func)