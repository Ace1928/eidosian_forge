import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_stack_in_progress(self):
    in_progress = self.stack.copy()
    in_progress['stack_status'] = 'CREATE_IN_PROGRESS'
    self.register_uris([dict(method='GET', uri='{endpoint}/stacks/{name}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name), status_code=302, headers=dict(location='{endpoint}/stacks/{name}/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name))), dict(method='GET', uri='{endpoint}/stacks/{name}/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name), json={'stack': in_progress})])
    res = self.cloud.get_stack(self.stack_name)
    self.assertIsNotNone(res)
    self.assertEqual(in_progress['stack_name'], res.name)
    self.assertEqual(in_progress['stack_status'], res['stack_status'])
    self.assertEqual('CREATE_IN_PROGRESS', res['status'])
    self.assert_calls()