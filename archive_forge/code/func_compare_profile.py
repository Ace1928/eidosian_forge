from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import \
def compare_profile(template, profile):
    diff = 0
    diff = diff + apply_diff_key(template, profile, ['BondingTechnology'])
    sip = profile.get('ServerInterfaceProfiles')
    for nic, ntw in sip.items():
        tmp = template.get(nic, {})
        diff = diff + apply_diff_key(tmp, ntw, ['NativeVLAN'])
        diff = diff + apply_diff_key(tmp, ntw, ['NicBonded'])
        untags = ntw.get('Networks')
        s = set(untags) | set(tmp.get('present', set()))
        s = s - set(tmp.get('absent', set()))
        if s.symmetric_difference(set(untags)):
            ntw['Networks'] = s
            diff = diff + 1
    return diff