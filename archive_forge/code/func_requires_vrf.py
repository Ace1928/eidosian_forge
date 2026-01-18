from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def requires_vrf(module, vrf):
    if not has_vrf(module, vrf):
        module.fail_json(msg='vrf %s is not configured' % vrf)