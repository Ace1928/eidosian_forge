from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def get_servers_to_add(current_servers, desired_servers):
    current_ids = [str(server['server_number']) for server in current_servers]
    current_ips = [server['server_ip'] for server in current_servers]
    current_ipv6s = [server['server_ipv6_net'] for server in current_servers]
    return [server for server in desired_servers if server not in current_ips and server not in current_ids and (server not in current_ipv6s)]