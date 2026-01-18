from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_order_with_octet_stream(self):
    content_type = 'application/octet-stream'
    self.props['payload_content_type'] = content_type
    defn = rsrc_defn.ResourceDefinition('foo', 'OS::Barbican::Order', self.props)
    res = self._create_resource(defn.name, defn, self.stack)
    args = self.barbican.orders.create.call_args[1]
    self.assertEqual(content_type, args[res.PAYLOAD_CONTENT_TYPE])