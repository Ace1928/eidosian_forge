import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test___delitem(self):
    del self.pgn[0]
    self.assertEqual(1, self.pgn.__len__())