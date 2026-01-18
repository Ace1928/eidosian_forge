from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def get_servers_to_delete(current_servers, desired_servers):
    return [server['server_ip'] for server in current_servers if server['server_ip'] not in desired_servers and server['server_ipv6_net'] not in desired_servers and (str(server['server_number']) not in desired_servers)]