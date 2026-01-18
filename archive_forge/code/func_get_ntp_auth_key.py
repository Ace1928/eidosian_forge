from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_ntp_auth_key(key_id, module):
    authentication_key = {}
    command = 'show run | inc ntp.authentication-key.{0}'.format(key_id)
    auth_regex = '.*ntp\\sauthentication-key\\s(?P<key_id>\\d+)\\smd5\\s(?P<md5string>\\S+)\\s(?P<atype>\\S+).*'
    body = execute_show_command(command, module)[0]
    try:
        match_authentication = re.match(auth_regex, body, re.DOTALL)
        group_authentication = match_authentication.groupdict()
        authentication_key['key_id'] = group_authentication['key_id']
        authentication_key['md5string'] = group_authentication['md5string']
        if group_authentication['atype'] == '7':
            authentication_key['auth_type'] = 'encrypt'
        else:
            authentication_key['auth_type'] = 'text'
    except (AttributeError, TypeError):
        authentication_key = {}
    return authentication_key