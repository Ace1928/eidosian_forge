from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def set_ntp_auth_key(key_id, md5string, auth_type, trusted_key, authentication):
    ntp_auth_cmds = []
    if key_id and md5string:
        auth_type_num = auth_type_to_num(auth_type)
        ntp_auth_cmds.append('ntp authentication-key {0} md5 {1} {2}'.format(key_id, md5string, auth_type_num))
    if trusted_key == 'true':
        ntp_auth_cmds.append('ntp trusted-key {0}'.format(key_id))
    elif trusted_key == 'false':
        ntp_auth_cmds.append('no ntp trusted-key {0}'.format(key_id))
    if authentication == 'on':
        ntp_auth_cmds.append('ntp authenticate')
    elif authentication == 'off':
        ntp_auth_cmds.append('no ntp authenticate')
    return ntp_auth_cmds