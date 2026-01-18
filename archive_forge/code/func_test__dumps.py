import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test__dumps(self):
    orig_dumps = json.dumps
    json.dumps = mock.Mock(return_value='test-dump')
    self.assertEqual('test-dump', self.cmd_base._dumps('item'))
    json.dumps = orig_dumps