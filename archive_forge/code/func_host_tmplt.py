from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def host_tmplt(config_data):
    host = config_data.get('host', '')
    command = 'snmp-server host {host}'.format(host=host)
    if config_data.get('informs'):
        command += ' informs'
    if config_data.get('traps'):
        command += ' traps'
    if config_data.get('version'):
        command += ' version {version}'.format(version=config_data['version'])
    if config_data.get('community'):
        command += ' {community}'.format(community=config_data['community'])
    if config_data.get('udp_port'):
        command += ' udp-port {udp_port}'.format(udp_port=config_data['udp_port'])
    if config_data.get('write'):
        command += ' write {write}'.format(write=config_data['write'])
    return command