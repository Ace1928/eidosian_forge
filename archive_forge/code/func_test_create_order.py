from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_order(self):
    res = self._create_resource('foo', self.res_template, self.stack)
    expected_state = (res.CREATE, res.COMPLETE)
    self.assertEqual(expected_state, res.state)
    args = self.barbican.orders.create.call_args[1]
    self.assertEqual('foobar-order', args['name'])
    self.assertEqual('aes', args['algorithm'])
    self.assertEqual('cbc', args['mode'])
    self.assertEqual(256, args['bit_length'])