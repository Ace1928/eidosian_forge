from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def get_powerflex_sds_parameters():
    """This method provide parameter required for the SDS module on
    PowerFlex"""
    return dict(sds_name=dict(), sds_id=dict(), sds_new_name=dict(), protection_domain_name=dict(), protection_domain_id=dict(), sds_ip_list=dict(type='list', elements='dict', options=dict(ip=dict(required=True), role=dict(required=True, choices=['all', 'sdsOnly', 'sdcOnly']))), sds_ip_state=dict(choices=['present-in-sds', 'absent-in-sds']), rfcache_enabled=dict(type='bool'), rmcache_enabled=dict(type='bool'), rmcache_size=dict(type='int'), performance_profile=dict(choices=['Compact', 'HighPerformance']), fault_set_name=dict(), fault_set_id=dict(), state=dict(required=True, type='str', choices=['present', 'absent']))