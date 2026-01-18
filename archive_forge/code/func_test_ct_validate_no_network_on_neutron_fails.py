from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_ct_validate_no_network_on_neutron_fails(self):
    self.t['resources']['cluster-template']['properties'].pop('neutron_management_network')
    ct = self._init_ct(self.t)
    self.patchobject(ct, 'is_using_neutron', return_value=True)
    ex = self.assertRaises(exception.StackValidationFailed, ct.validate)
    self.assertEqual('neutron_management_network must be provided', str(ex))