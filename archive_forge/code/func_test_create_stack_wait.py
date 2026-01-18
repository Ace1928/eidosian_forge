import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_stack_wait(self):
    test_template = tempfile.NamedTemporaryFile(delete=False)
    test_template.write(fakes.FAKE_TEMPLATE.encode('utf-8'))
    test_template.close()
    self.register_uris([dict(method='POST', uri='{endpoint}/stacks'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT), json={'stack': self.stack}, validate=dict(json={'disable_rollback': False, 'parameters': {}, 'stack_name': self.stack_name, 'tags': self.stack_tag, 'template': fakes.FAKE_TEMPLATE_CONTENT, 'timeout_mins': 60})), dict(method='GET', uri='{endpoint}/stacks/{name}/events?sort_dir=asc'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name), json={'events': [fakes.make_fake_stack_event(self.stack_id, self.stack_name, status='CREATE_COMPLETE', resource_name='name')]}), dict(method='GET', uri='{endpoint}/stacks/{name}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name), status_code=302, headers=dict(location='{endpoint}/stacks/{name}/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name))), dict(method='GET', uri='{endpoint}/stacks/{name}/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name), json={'stack': self.stack})])
    self.cloud.create_stack(self.stack_name, tags=self.stack_tag, template_file=test_template.name, wait=True)
    self.assert_calls()