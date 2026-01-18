import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_stack_not_found(self):
    resolve = 'resolve_outputs=False'
    self.register_uris([dict(method='GET', uri='{endpoint}/stacks/stack_name?{resolve}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, resolve=resolve), status_code=404)])
    self.assertFalse(self.cloud.delete_stack('stack_name'))
    self.assert_calls()