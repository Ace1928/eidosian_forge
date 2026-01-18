from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_replication_session_parameters():
    """This method provide parameters required for the ansible replication session
       module on Unity"""
    return dict(session_id=dict(type='str'), session_name=dict(type='str'), pause=dict(type='bool'), sync=dict(type='bool'), force=dict(type='bool'), failover_with_sync=dict(type='bool'), failback=dict(type='bool'), force_full_copy=dict(type='bool'), state=dict(type='str', choices=['present', 'absent'], default='present'))