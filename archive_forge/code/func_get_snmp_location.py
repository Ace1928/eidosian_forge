from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_snmp_location(module):
    location = {}
    location_regex = '^\\s*snmp-server\\s+location\\s+(?P<location>.+)$'
    body = execute_show_command('show run snmp', module)[0]
    match_location = re.search(location_regex, body, re.M)
    if match_location:
        location['location'] = match_location.group('location')
    return location