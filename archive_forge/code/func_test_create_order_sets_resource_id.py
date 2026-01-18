from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_order_sets_resource_id(self):
    self.barbican.orders.create.return_value = FakeOrder('foo')
    res = self._create_resource('foo', self.res_template, self.stack)
    self.assertEqual('foo', res.resource_id)