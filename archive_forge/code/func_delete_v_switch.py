from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def delete_v_switch(module, id_):
    url = '{0}/{1}'.format(V_SWITCH_BASE_URL, id_)
    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    data = {'cancellation_date': datetime.now().strftime('%y-%m-%d')}
    result, error = fetch_url_json(module, url, data=urlencode(data), headers=headers, method='DELETE', accept_errors=['INVALID_INPUT', 'NOT_FOUND', 'CONFLICT'], allow_empty_result=True)
    if error == 'INVALID_INPUT':
        invalid_parameters = print_list(result['error']['invalid'])
        module.fail_json(msg='vSwitch invalid parameter ({0})'.format(invalid_parameters))
    elif error == 'NOT_FOUND':
        module.fail_json(msg='vSwitch not found to delete')
    elif error == 'CONFLICT':
        module.fail_json(msg='The vSwitch is already cancelled')
    return result