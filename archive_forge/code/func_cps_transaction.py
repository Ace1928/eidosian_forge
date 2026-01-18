from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import dict_diff
def cps_transaction(obj):
    RESULT = dict()
    ch = {'operation': obj.get_property('oper'), 'change': obj.get()}
    if cps.transaction([ch]):
        RESULT['response'] = convert_cps_raw_list([ch['change']])
        RESULT['changed'] = True
    else:
        error_msg = 'Transaction error while ' + obj.get_property('oper')
        raise RuntimeError(error_msg)
    return RESULT