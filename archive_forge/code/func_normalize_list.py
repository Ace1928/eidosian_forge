from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.urls import build_service_uri
from ..module_utils.teem import send_teem
def normalize_list(self, lists):
    result = []
    for list in lists:
        tmp = dict(((str(k), str(v)) for k, v in iteritems(list) if k != 'value'))
        if 'encrypted' not in list:
            tmp['encrypted'] = 'no'
        if 'value' in list:
            if len(list['value']) > 0:
                tmp['value'] = [str(x) for x in list['value']]
        if tmp['encrypted'] == 'True':
            tmp['encrypted'] = 'yes'
        elif tmp['encrypted'] == 'False':
            tmp['encrypted'] = 'no'
        elif isinstance(tmp['encrypted'], bool):
            if tmp['encrypted'] is True:
                tmp['encrypted'] = 'yes'
            else:
                tmp['encrypted'] = 'no'
        result.append(tmp)
    result = sorted(result, key=lambda k: k['name'])
    return result