from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def delete_servers(module, id_, servers):
    url = '{0}/{1}/server'.format(V_SWITCH_BASE_URL, id_)
    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    data = get_x_www_form_urlenconded_dict_from_list('server', servers)
    result, error = fetch_url_json(module, url, data=urlencode(data), headers=headers, method='DELETE', accept_errors=['SERVER_NOT_FOUND', 'VSWITCH_IN_PROCESS'], allow_empty_result=True)
    if error == 'SERVER_NOT_FOUND':
        module.fail_json(msg=result['error']['message'])
    elif error == 'VSWITCH_IN_PROCESS':
        module.fail_json(msg='There is a update running, therefore the vswitch can not be updated')
    wait_condition = is_all_servers_ready if module.params['wait'] else None
    return get_v_switch(module, id_, wait_condition)