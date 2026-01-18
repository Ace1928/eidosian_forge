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
def test_loadbalancer_attr_dnsname(self):
    rsrc = self.setup_loadbalancer()
    rsrc.get_output = mock.Mock(return_value='1.3.5.7')
    self.assertEqual('1.3.5.7', rsrc.FnGetAtt('DNSName'))
    rsrc.get_output.assert_called_once_with('PublicIp')