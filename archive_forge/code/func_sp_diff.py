from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from urllib.parse import quote
def sp_diff(want, have):
    hsps = {}
    wsps = {}
    dsps = []
    for hsp in have.get('security_profiles') or []:
        hsps[hsp.get('profile_name')] = hsp
    for wsp in want.get('security_profiles') or []:
        wsps[wsp.get('profile_name')] = wsp
    for spn, sp in wsps.items():
        dsp = dict(hsps.get(spn))
        for k, v in dsp.items():
            if not isinstance(dsp.get(k), list) and (not isinstance(dsp.get(k), dict)):
                if k not in sp:
                    dsp.pop(k)
        for k, v in sp.items():
            if not isinstance(dsp.get(k), list) and (not isinstance(dsp.get(k), dict)):
                if dsp.get(k) != v:
                    dsp[k] = v
            elif v is not None:
                dsp[k] = v
        if dsp != hsps.get(spn):
            dsps.append(dsp)
    return dsps