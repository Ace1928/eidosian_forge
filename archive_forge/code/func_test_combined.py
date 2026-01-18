import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_combined(self):
    self.lb._haproxy_config_global = mock.Mock(return_value='one,')
    self.lb._haproxy_config_frontend = mock.Mock(return_value='two,')
    self.lb._haproxy_config_backend = mock.Mock(return_value='three,')
    self.lb._haproxy_config_servers = mock.Mock(return_value='four')
    actual = self.lb._haproxy_config([3, 5])
    self.assertEqual('one,two,three,four\n', actual)
    self.lb._haproxy_config_global.assert_called_once_with()
    self.lb._haproxy_config_frontend.assert_called_once_with()
    self.lb._haproxy_config_backend.assert_called_once_with()
    self.lb._haproxy_config_servers.assert_called_once_with([3, 5])