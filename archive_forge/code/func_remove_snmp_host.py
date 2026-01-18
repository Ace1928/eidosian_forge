from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def remove_snmp_host(host, udp, existing):
    commands = []
    if existing['version'] == 'v3':
        existing['version'] = '3'
        command = 'no snmp-server host {0} {snmp_type} version                     {version} {v3} {community} udp-port {1}'.format(host, udp, **existing)
    elif existing['version'] == 'v2c':
        existing['version'] = '2c'
        command = 'no snmp-server host {0} {snmp_type} version                     {version} {community} udp-port {1}'.format(host, udp, **existing)
    elif existing['version'] == 'v1':
        existing['version'] = '1'
        command = 'no snmp-server host {0} {snmp_type} version                     {version} {community} udp-port {1}'.format(host, udp, **existing)
    if command:
        commands.append(command)
    return commands