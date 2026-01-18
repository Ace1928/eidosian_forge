from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import container
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_generic(self):
    res = self._create_resource('foo')
    expected_state = (res.CREATE, res.COMPLETE)
    self.assertEqual(expected_state, res.state)
    args = self.client_plugin.create_generic_container.call_args[1]
    self.assertEqual('mynewcontainer', args['name'])
    self.assertEqual({'secret1': 'ref1', 'secret2': 'ref2'}, args['secret_refs'])
    self.assertEqual(sorted(['ref1', 'ref2']), sorted(res.get_refs()))