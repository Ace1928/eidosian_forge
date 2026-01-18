from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def get_v_switch(module, id_, wait_condition=None):
    url = '{0}/{1}'.format(V_SWITCH_BASE_URL, id_)
    accept_errors = ['NOT_FOUND']
    if wait_condition:
        try:
            result, error = fetch_url_json_with_retries(module, url, check_done_callback=wait_condition, check_done_delay=module.params['wait_delay'], check_done_timeout=module.params['timeout'], accept_errors=accept_errors)
        except CheckDoneTimeoutException as dummy:
            module.fail_json(msg='Timeout waiting vSwitch operation to finish')
    else:
        result, error = fetch_url_json(module, url, accept_errors=accept_errors)
    if error == 'NOT_FOUND':
        module.fail_json(msg='vSwitch not found.')
    return result