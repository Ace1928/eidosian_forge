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
def test_endpoint_handle_update_only_enabled(self):
    rsrc = self._setup_endpoint_resource('test_endpoint_update_enabled')
    rsrc.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {endpoint.KeystoneEndpoint.ENABLED: True}
    rsrc.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.endpoints.update.assert_called_once_with(endpoint=rsrc.resource_id, region=None, interface=None, service=None, url=None, name=None, enabled=prop_diff[endpoint.KeystoneEndpoint.ENABLED])