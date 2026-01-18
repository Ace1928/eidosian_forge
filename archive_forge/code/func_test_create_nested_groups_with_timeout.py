import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_create_nested_groups_with_timeout(self):
    parent_template = '\nheat_template_version: rocky\nresources:\n  parent_group:\n    type: OS::Heat::ResourceGroup\n    update_policy:\n      batch_create: { max_batch_size: 1, pause_time: 1 }\n    properties:\n      count: 2\n      resource_def:\n        type: child.yaml\n'
    child_template = '\nheat_template_version: rocky\nresources:\n  child_group:\n    type: OS::Heat::ResourceGroup\n    update_policy:\n      batch_create: { max_batch_size: 1, pause_time: 1 }\n    properties:\n      count: 2\n      resource_def:\n        type: value.yaml\n'
    value_template = "\nheat_template_version: rocky\nresources:\n  value:\n    type: OS::Heat::Value\n    properties:\n      type: string\n      value: 'test'\n"
    files = {'child.yaml': child_template, 'value.yaml': value_template}
    stack_identifier = self.stack_create(template=parent_template, files=files, timeout=10)
    resources = self.client.resources.list(stack_identifier, nested_depth=2, with_detail=True)
    timeouts = set()
    for res in resources:
        if res.resource_type == 'OS::Heat::ResourceGroup':
            nested_stack = self.client.stacks.get(res.physical_resource_id)
            timeouts.add(nested_stack.timeout_mins)
    self.assertEqual({10}, timeouts)