from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_diff_aaa(self, want, have):
    diff_cfg = {}
    diff_authentication = {}
    diff_data = {}
    authentication = want.get('authentication', None)
    if authentication:
        data = authentication.get('data', None)
        if data:
            fail_through = data.get('fail_through', None)
            local = data.get('local', None)
            group = data.get('group', None)
            cfg_authentication = have.get('authentication', None)
            if cfg_authentication:
                cfg_data = cfg_authentication.get('data', None)
                if cfg_data:
                    cfg_fail_through = cfg_data.get('fail_through', None)
                    cfg_local = cfg_data.get('local', None)
                    cfg_group = cfg_data.get('group', None)
                    if fail_through is not None and fail_through != cfg_fail_through:
                        diff_data['fail_through'] = fail_through
                    if local and local != cfg_local:
                        diff_data['local'] = local
                    if group and group != cfg_group:
                        diff_data['group'] = group
                    diff_local = diff_data.get('local', None)
                    diff_group = diff_data.get('group', None)
                    if diff_local and (not diff_group) and cfg_group:
                        diff_data['group'] = cfg_group
                    if diff_group and (not diff_local) and cfg_local:
                        diff_data['local'] = cfg_local
            else:
                if fail_through is not None:
                    diff_data['fail_through'] = fail_through
                if local:
                    diff_data['local'] = local
                if group:
                    diff_data['group'] = group
            if diff_data:
                diff_authentication['data'] = diff_data
                diff_cfg['authentication'] = diff_authentication
    return diff_cfg