from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def properties_facts(self):
    ansible_facts = self.to_json(self.host, self.params.get('properties'))
    if self.params.get('show_tag'):
        vmware_client = VmwareRestClient(self.module)
        tag_info = {'tags': vmware_client.get_tags_for_hostsystem(hostsystem_mid=self.host._moId)}
        ansible_facts.update(tag_info)
    self.module.exit_json(changed=False, ansible_facts=ansible_facts)