from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def get_powerflex_volume_parameters():
    """This method provide parameter required for the volume
    module on PowerFlex"""
    return dict(vol_name=dict(), vol_id=dict(), storage_pool_name=dict(), storage_pool_id=dict(), protection_domain_name=dict(), protection_domain_id=dict(), use_rmcache=dict(type='bool'), snapshot_policy_name=dict(), snapshot_policy_id=dict(), size=dict(type='int'), cap_unit=dict(choices=['GB', 'TB']), vol_type=dict(choices=['THICK_PROVISIONED', 'THIN_PROVISIONED']), compression_type=dict(choices=['NORMAL', 'NONE']), auto_snap_remove_type=dict(choices=['detach', 'remove']), vol_new_name=dict(), allow_multiple_mappings=dict(type='bool'), delete_snapshots=dict(type='bool'), sdc=dict(type='list', elements='dict', options=dict(sdc_id=dict(), sdc_ip=dict(), sdc_name=dict(), access_mode=dict(choices=['READ_WRITE', 'READ_ONLY', 'NO_ACCESS']), bandwidth_limit=dict(type='int'), iops_limit=dict(type='int'))), sdc_state=dict(choices=['mapped', 'unmapped']), state=dict(required=True, type='str', choices=['present', 'absent']))