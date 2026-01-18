from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import secret
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_secret(self):
    expected_state = (self.res.CREATE, self.res.COMPLETE)
    self.assertEqual(expected_state, self.res.state)
    args = self.barbican.secrets.create.call_args[1]
    self.assertEqual('foobar-secret', args['name'])
    self.assertEqual('opaque', args['secret_type'])