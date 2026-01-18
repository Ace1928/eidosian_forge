from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import container
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_certificate(self):
    res = self._create_resource('foo', tmpl_name='OS::Barbican::CertificateContainer')
    expected_state = (res.CREATE, res.COMPLETE)
    self.assertEqual(expected_state, res.state)
    args = self.client_plugin.create_certificate.call_args[1]
    self.assertEqual('mynewcontainer', args['name'])
    self.assertEqual('cref', args['certificate_ref'])
    self.assertEqual('pkref', args['private_key_ref'])
    self.assertEqual('pkpref', args['private_key_passphrase_ref'])
    self.assertEqual('iref', args['intermediates_ref'])
    self.assertEqual(sorted(['pkref', 'pkpref', 'iref', 'cref']), sorted(res.get_refs()))