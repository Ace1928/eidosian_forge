import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_eip(self):
    mock_server = self.fc.servers.list()[0]
    self.patchobject(self.fc.servers, 'get', return_value=mock_server)
    self.mock_create_floatingip()
    iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
    self.patchobject(mock_server, 'interface_list', return_value=[iface])
    t = template_format.parse(eip_template)
    stack = utils.parse_stack(t)
    rsrc = self.create_eip(t, stack, 'IPAddress')
    try:
        self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
        rsrc.refid = None
        self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
        self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', rsrc.FnGetAtt('AllocationId'))
        self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'Foo')
    finally:
        scheduler.TaskRunner(rsrc.destroy)()