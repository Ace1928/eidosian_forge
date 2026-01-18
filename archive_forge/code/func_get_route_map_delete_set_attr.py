from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_set_attr(self, command, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST APIs needed
        for deletion of all eligible "set" attributes contained in the
        user input command dict specified by the "command" input parameter
        to this function. Modify the contents of the "command" object to
        remove any attributes that are not currently configured. These
        attributes are not "eligible" for deletion and no REST API "request"
        is generated for them."""
    cmd_set_top = command.get('set')
    if not cmd_set_top:
        return
    set_keys = cmd_set_top.keys()
    cfg_set_top = cmd_rmap_have.get('set')
    if not cfg_set_top:
        command.pop('set')
        return
    cfg_set_keys = cfg_set_top.keys()
    set_both_keys = set(set_keys).intersection(cfg_set_keys)
    if not set_both_keys:
        command.pop('set')
        return
    conf_map_name = command['map_name']
    conf_seq_num = command['sequence_num']
    req_seq_num = str(conf_seq_num)
    set_delete_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'actions/'
    self.get_route_map_delete_set_bgp(command, set_both_keys, cmd_rmap_have, requests)
    cmd_set_top = command.get('set')
    if not cmd_set_top:
        command.pop('set')
        return
    if 'metric' in set_both_keys:
        set_delete_metric_base = set_delete_base + 'metric-action/config'
        if cmd_set_top['metric'].get('rtt_action'):
            if cmd_set_top['metric']['rtt_action'] == cfg_set_top['metric'].get('rtt_action'):
                request_uri = set_delete_metric_base
                request = {'path': request_uri, 'method': DELETE}
                requests.append(request)
            else:
                cmd_set_top.pop('metric')
                if not cmd_set_top:
                    command.pop('set')
        elif cmd_set_top['metric'].get('value'):
            set_delete_bgp_base = set_delete_base + 'openconfig-bgp-policy:bgp-actions/'
            if cmd_set_top['metric']['value'] == cfg_set_top['metric'].get('value'):
                request = {'path': set_delete_metric_base, 'method': DELETE}
                requests.append(request)
                request = {'path': set_delete_bgp_base + 'config/set-med', 'method': DELETE}
                requests.append(request)
            else:
                cmd_set_top.pop('metric')
                if not cmd_set_top:
                    command.pop('set')
    elif cmd_set_top.get('metric'):
        cmd_set_top.pop('metric')
        if not cmd_set_top:
            command.pop('set')
            return