import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_gateway_router(self):

    def find_rsrc(resource, name_or_id, cmd_resource=None):
        id_mapping = {'router_id': '3e46229d-8fce-4733-819a-b5fe630550f8', 'network': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}
        return id_mapping.get(resource)
    self.find_rsrc_mock.side_effect = find_rsrc
    self.remove_gw_mock.side_effect = [None, qe.NeutronClientException(status_code=404)]
    self.stub_RouterConstraint_validate()
    t = template_format.parse(neutron_template)
    stack = utils.parse_stack(t)
    rsrc = self.create_gateway_router(t, stack, 'gateway', properties={'router_id': '3e46229d-8fce-4733-819a-b5fe630550f8', 'network': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'})
    self.add_gw_mock.assert_called_with('3e46229d-8fce-4733-819a-b5fe630550f8', {'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'})
    scheduler.TaskRunner(rsrc.delete)()
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    scheduler.TaskRunner(rsrc.delete)()