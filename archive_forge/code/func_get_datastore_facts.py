from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_datastore_facts(self):
    facts = dict()
    facts['ansible_datastore'] = []
    for store in self.host.datastore:
        _tmp = {'name': store.summary.name, 'total': bytes_to_human(store.summary.capacity), 'free': bytes_to_human(store.summary.freeSpace)}
        facts['ansible_datastore'].append(_tmp)
    return facts