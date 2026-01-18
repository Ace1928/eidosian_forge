from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
def test_list_nested(self):
    self._test_list(fields={'stack_id': 'teststack', 'nested_depth': '99'}, expect='/stacks/teststack/resources?%s' % parse.urlencode({'nested_depth': 99}, True))