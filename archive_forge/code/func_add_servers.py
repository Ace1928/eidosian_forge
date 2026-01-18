from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def add_servers(module, id_, servers):
    url = '{0}/{1}/server'.format(V_SWITCH_BASE_URL, id_)
    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    data = get_x_www_form_urlenconded_dict_from_list('server', servers)
    result, error = fetch_url_json(module, url, data=urlencode(data), headers=headers, method='POST', accept_errors=['INVALID_INPUT', 'SERVER_NOT_FOUND', 'VSWITCH_VLAN_NOT_UNIQUE', 'VSWITCH_IN_PROCESS', 'VSWITCH_SERVER_LIMIT_REACHED'], allow_empty_result=True, allowed_empty_result_status_codes=(201,))
    if error == 'INVALID_INPUT':
        invalid_parameters = print_list(result['error']['invalid'])
        module.fail_json(msg='Invalid parameter adding server ({0})'.format(invalid_parameters))
    elif error == 'SERVER_NOT_FOUND':
        module.fail_json(msg=result['error']['message'])
    elif error == 'VSWITCH_VLAN_NOT_UNIQUE':
        module.fail_json(msg=result['error']['message'])
    elif error == 'VSWITCH_IN_PROCESS':
        module.fail_json(msg='There is a update running, therefore the vswitch can not be updated')
    elif error == 'VSWITCH_SERVER_LIMIT_REACHED':
        module.fail_json(msg='The maximum number of servers is reached for this vSwitch')
    wait_condition = is_all_servers_ready if module.params['wait'] else None
    return get_v_switch(module, id_, wait_condition)