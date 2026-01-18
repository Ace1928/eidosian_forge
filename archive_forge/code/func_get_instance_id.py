from __future__ import absolute_import, division, print_function
import urllib
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
def get_instance_id(self):
    instance_id = self.module.params['instance_id']
    if instance_id is not None:
        return instance_id
    instance_name = self.module.params['instance_name']
    if instance_name is not None:
        if len(instance_name) == 0:
            return ''
        try:
            label = urllib.quote(instance_name)
        except AttributeError:
            label = urllib.parse.quote(instance_name)
        resources = self.api_query(path='/instances?label=%s' % label) or dict()
        if not resources or not resources['instances']:
            self.module.fail_json(msg='No instance with name found: %s' % instance_name)
        if len(resources['instances']) > 1:
            self.module.fail_json(msg='More then one instance with name found: %s' % instance_name)
        return resources['instances'][0]['id']