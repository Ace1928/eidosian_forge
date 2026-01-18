from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import subnetpool
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_create_subnetpool(self):
    rsrc = self.create_subnetpool()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', ref_id)