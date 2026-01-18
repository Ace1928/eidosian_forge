from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import dict_diff
def parse_key_data(attrs):
    res = dict()
    for key, val in iteritems(attrs):
        if key == 'cps/key_data':
            res.update(val)
        else:
            res[key] = val
    return res