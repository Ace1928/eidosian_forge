import hashlib
import json
import random
from urllib import parse
from swiftclient import utils as swiftclient_utils
import yaml
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_nested_stack_adopt_ok(self):
    url = self.publish_template(self.nested_with_res_template)
    self.template = self.test_template.replace('the.yaml', url)
    adopt_data = {'resources': {'the_nested': {'resource_id': 'test-res-id', 'resources': {'NestedResource': {'type': 'OS::Heat::RandomString', 'resource_id': 'test-nested-res-id', 'resource_data': {'value': 'goopie'}}}}}, 'environment': {'parameters': {}}, 'template': yaml.safe_load(self.template)}
    stack_identifier = self.stack_adopt(adopt_data=json.dumps(adopt_data))
    self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
    stack = self.client.stacks.get(stack_identifier)
    self.assertEqual('goopie', self._stack_output(stack, 'output_foo'))