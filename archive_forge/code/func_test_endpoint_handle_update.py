import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import endpoint
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_endpoint_handle_update(self):
    rsrc = self._setup_endpoint_resource('test_endpoint_update')
    rsrc.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {endpoint.KeystoneEndpoint.REGION: 'RegionTwo', endpoint.KeystoneEndpoint.INTERFACE: 'internal', endpoint.KeystoneEndpoint.SERVICE: 'updated_id', endpoint.KeystoneEndpoint.SERVICE_URL: 'http://127.0.0.1:8004/v2/tenant-id', endpoint.KeystoneEndpoint.NAME: 'endpoint_foo_updated', endpoint.KeystoneEndpoint.ENABLED: True}
    rsrc.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.endpoints.update.assert_called_once_with(endpoint=rsrc.resource_id, region=prop_diff[endpoint.KeystoneEndpoint.REGION], interface=prop_diff[endpoint.KeystoneEndpoint.INTERFACE], service=prop_diff[endpoint.KeystoneEndpoint.SERVICE], url=prop_diff[endpoint.KeystoneEndpoint.SERVICE_URL], name=prop_diff[endpoint.KeystoneEndpoint.NAME], enabled=prop_diff[endpoint.KeystoneEndpoint.ENABLED])