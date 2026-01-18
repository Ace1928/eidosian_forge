import copy
from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception as heat_ex
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.nova import floatingip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def create_delete_assoc_with_exc(self, exc):
    rsrc = self.prepare_floating_ip_assoc()
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.mock_upd_fip.side_effect = [exc]
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)