from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acls.acls import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def sanitize_data(self, data):
    """removes matches or extra config info that is added on acl match"""
    re_data = ''
    remarks_idx = 0
    for da in data.split('\n'):
        if 'match' in da:
            mod_da = re.sub('\\([^()]*\\)', '', da)
            re_data += mod_da[:-1] + '\n'
        elif re.match('\\s*\\d+\\sremark.+', da, re.IGNORECASE) or re.match('\\s*remark.+', da, re.IGNORECASE):
            remarks_idx += 1
            re_data += to_text(remarks_idx) + ' ' + da + '\n'
        else:
            re_data += da + '\n'
    return re_data