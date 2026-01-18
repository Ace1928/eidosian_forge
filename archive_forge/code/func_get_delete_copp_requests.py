from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_copp_requests(self, commands, have, is_delete_all):
    requests = []
    if is_delete_all:
        copp_groups = commands.get('copp_groups', None)
        if copp_groups:
            for group in copp_groups:
                copp_name = group.get('copp_name', None)
                requests.append(self.get_delete_single_copp_group_request(copp_name))
    else:
        copp_groups = commands.get('copp_groups', None)
        if copp_groups:
            for group in copp_groups:
                copp_name = group.get('copp_name', None)
                trap_priority = group.get('trap_priority', None)
                trap_action = group.get('trap_action', None)
                queue = group.get('queue', None)
                cir = group.get('cir', None)
                cbs = group.get('cbs', None)
                if have:
                    cfg_copp_groups = have.get('copp_groups', None)
                    if cfg_copp_groups:
                        for cfg_group in cfg_copp_groups:
                            cfg_copp_name = cfg_group.get('copp_name', None)
                            cfg_trap_priority = cfg_group.get('trap_priority', None)
                            cfg_trap_action = cfg_group.get('trap_action', None)
                            cfg_queue = cfg_group.get('queue', None)
                            cfg_cir = cfg_group.get('cir', None)
                            cfg_cbs = cfg_group.get('cbs', None)
                            if copp_name == cfg_copp_name:
                                if trap_priority and trap_priority == cfg_trap_priority:
                                    requests.append(self.get_delete_copp_groups_attr_request(copp_name, 'trap-priority'))
                                if trap_action and trap_action == cfg_trap_action:
                                    err_msg = 'Deletion of trap-action attribute is not supported.'
                                    self._module.fail_json(msg=err_msg, code=405)
                                    requests.append(self.get_delete_copp_groups_attr_request(copp_name, 'trap-action'))
                                if queue and queue == cfg_queue:
                                    requests.append(self.get_delete_copp_groups_attr_request(copp_name, 'queue'))
                                if cir and cir == cfg_cir:
                                    requests.append(self.get_delete_copp_groups_attr_request(copp_name, 'cir'))
                                if cbs and cbs == cfg_cbs:
                                    requests.append(self.get_delete_copp_groups_attr_request(copp_name, 'cbs'))
                                if not trap_priority and (not trap_action) and (not queue) and (not cir) and (not cbs):
                                    requests.append(self.get_delete_single_copp_group_request(copp_name))
    return requests