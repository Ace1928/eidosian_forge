import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_make_size_human_readable(self):
    self.assertEqual('106B', utils.make_size_human_readable(106))
    self.assertEqual('1000kB', utils.make_size_human_readable(1024000))
    self.assertEqual('1MB', utils.make_size_human_readable(1048576))
    self.assertEqual('1.4GB', utils.make_size_human_readable(1476395008))
    self.assertEqual('9.3MB', utils.make_size_human_readable(9761280))
    self.assertEqual('0B', utils.make_size_human_readable(None))