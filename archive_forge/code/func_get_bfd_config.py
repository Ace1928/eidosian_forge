from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bfd.bfd import BfdArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_bfd_config(self, module):
    bfd_cfg = None
    get_bfd_path = '/data/openconfig-bfd:bfd'
    request = {'path': get_bfd_path, 'method': 'get'}
    try:
        response = edit_config(module, to_request(module, request))
        if 'openconfig-bfd:bfd' in response[0][1]:
            bfd_cfg = response[0][1].get('openconfig-bfd:bfd', None)
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    return bfd_cfg