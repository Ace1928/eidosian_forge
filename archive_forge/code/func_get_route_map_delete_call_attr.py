from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_call_attr(self, command, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST API needed
        for deletion of the "call" attribute if this attribute it contained in the
        user input command dict specified by the "command" input parameter
        to this function and it is currently configured. Modify the contents of
        the "command" object to remove the "call" attribute if it is not currently
        configured."""
    if not command.get('call'):
        return
    if not command['call'] == cmd_rmap_have.get('call'):
        command.pop('call')
        return
    conf_map_name = command['map_name']
    req_seq_num = str(command['sequence_num'])
    call_delete_req_uri = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'conditions/config/call-policy'
    request = {'path': call_delete_req_uri, 'method': DELETE}
    requests.append(request)