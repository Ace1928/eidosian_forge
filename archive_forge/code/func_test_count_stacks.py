import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_count_stacks(self):
    default_args = {'filters': mock.ANY, 'show_deleted': mock.ANY, 'show_nested': mock.ANY, 'show_hidden': mock.ANY, 'tags': mock.ANY, 'tags_any': mock.ANY, 'not_tags': mock.ANY, 'not_tags_any': mock.ANY}
    self._test_engine_api('count_stacks', 'call', **default_args)