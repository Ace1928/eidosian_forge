from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_powerflex_info_parameters():
    """This method provides parameters required for the ansible
    info module on powerflex"""
    return dict(gather_subset=dict(type='list', required=False, elements='str', choices=['vol', 'storage_pool', 'protection_domain', 'sdc', 'sds', 'snapshot_policy', 'device', 'rcg', 'replication_pair', 'fault_set', 'service_template', 'managed_device', 'deployment']), filters=dict(type='list', required=False, elements='dict', options=dict(filter_key=dict(type='str', required=True, no_log=False), filter_operator=dict(type='str', required=True, choices=['equal', 'contains']), filter_value=dict(type='str', required=True))), sort=dict(type='str'), limit=dict(type='int', default=50), offset=dict(type='int', default=0), include_devices=dict(type='bool', default=True), include_template=dict(type='bool', default=True), full=dict(type='bool', default=False), include_attachments=dict(type='bool', default=True))