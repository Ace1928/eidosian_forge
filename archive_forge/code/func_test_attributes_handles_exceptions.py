from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import secret
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_attributes_handles_exceptions(self):
    self.barbican.barbican_client.HTTPClientError = Exception
    self.barbican.secrets.get.side_effect = Exception('boom')
    self.assertRaises(self.barbican.barbican_client.HTTPClientError, self.res.FnGetAtt, 'order_ref')