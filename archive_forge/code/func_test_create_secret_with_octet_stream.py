from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import secret
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_secret_with_octet_stream(self):
    content_type = 'application/octet-stream'
    props = {'name': 'secret', 'payload': 'foobar', 'payload_content_type': content_type}
    defn = rsrc_defn.ResourceDefinition('secret', 'OS::Barbican::Secret', props)
    res = self._create_resource(defn.name, defn, self.stack)
    args = self.barbican.secrets.create.call_args[1]
    self.assertEqual('foobar', args[res.PAYLOAD])
    self.assertEqual(content_type, args[res.PAYLOAD_CONTENT_TYPE])