from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def get_failover(module, ip):
    """
    Get current routing target of failover IP.

    The value ``None`` represents unrouted.

    See https://robot.your-server.de/doc/webservice/en.html#get-failover-failover-ip
    """
    return get_failover_record(module, ip)['active_server_ip']