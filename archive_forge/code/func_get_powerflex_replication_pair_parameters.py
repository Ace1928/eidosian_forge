from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_powerflex_replication_pair_parameters():
    """This method provide parameter required for the replication_consistency_group
    module on PowerFlex"""
    return dict(pair_id=dict(), pair_name=dict(), pause=dict(type='bool'), state=dict(choices=['absent', 'present'], default='present'), rcg_id=dict(), rcg_name=dict(), remote_peer=dict(type='dict', options=dict(hostname=dict(type='str', aliases=['gateway_host'], required=True), username=dict(type='str', required=True), password=dict(type='str', required=True, no_log=True), validate_certs=dict(type='bool', aliases=['verifycert'], default=True), port=dict(type='int', default=443), timeout=dict(type='int', default=120))), pairs=dict(type='list', elements='dict', options=dict(source_volume_name=dict(), source_volume_id=dict(), target_volume_name=dict(), target_volume_id=dict(), copy_type=dict(required=True, choices=['Identical', 'OnlineCopy', 'OnlineHashCopy', 'OfflineCopy']), name=dict())))