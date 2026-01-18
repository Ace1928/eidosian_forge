import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test___reversed__(self):
    itr = self.pgn.__reversed__()
    self.assertEqual('item2', next(itr))
    self.assertEqual('item1', next(itr))
    self.assertRaises(StopIteration, next, itr)