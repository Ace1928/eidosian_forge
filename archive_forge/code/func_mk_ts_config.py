from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from urllib.parse import quote
def mk_ts_config(indata):
    outdata = {k.replace('_', '-'): v for k, v in indata.items() if v is not None}
    output = {'openconfig-pki:trust-store': [{'name': outdata.get('name'), 'config': outdata}]}
    return output