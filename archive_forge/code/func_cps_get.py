from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import dict_diff
def cps_get(obj):
    RESULT = dict()
    key = obj.get()
    l = []
    cps.get([key], l)
    resp_list = convert_cps_raw_list(l)
    RESULT['response'] = resp_list
    return RESULT