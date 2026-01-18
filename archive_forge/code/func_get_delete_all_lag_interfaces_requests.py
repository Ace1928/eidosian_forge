from __future__ import absolute_import, division, print_function
import json
from copy import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def get_delete_all_lag_interfaces_requests(self):
    requests = []
    delete_all_lag_url = 'data/sonic-portchannel:sonic-portchannel/PORTCHANNEL_MEMBER/PORTCHANNEL_MEMBER_LIST'
    method = DELETE
    delete_all_lag_request = {'path': delete_all_lag_url, 'method': method}
    requests.append(delete_all_lag_request)
    return requests