from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import container
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_rsa(self):
    res = self._create_resource('foo', tmpl_name='OS::Barbican::RSAContainer')
    expected_state = (res.CREATE, res.COMPLETE)
    self.assertEqual(expected_state, res.state)
    args = self.client_plugin.create_rsa.call_args[1]
    self.assertEqual('mynewcontainer', args['name'])
    self.assertEqual('pkref', args['private_key_ref'])
    self.assertEqual('pubref', args['public_key_ref'])
    self.assertEqual('pkpref', args['private_key_passphrase_ref'])
    self.assertEqual(sorted(['pkref', 'pubref', 'pkpref']), sorted(res.get_refs()))