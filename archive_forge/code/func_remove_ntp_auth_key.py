from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def remove_ntp_auth_key(key_id, md5string, auth_type, trusted_key, authentication):
    auth_remove_cmds = []
    if key_id:
        auth_type_num = auth_type_to_num(auth_type)
        auth_remove_cmds.append('no ntp authentication-key {0} md5 {1} {2}'.format(key_id, md5string, auth_type_num))
    if authentication:
        auth_remove_cmds.append('no ntp authenticate')
    return auth_remove_cmds