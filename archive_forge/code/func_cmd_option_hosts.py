from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def cmd_option_hosts(config_data):
    cmd = ''
    if config_data:
        cmd = 'snmp-server host'
        if config_data.get('host'):
            cmd += ' {host}'.format(host=config_data.get('host'))
        if config_data.get('informs'):
            cmd += ' informs'
        if config_data.get('version'):
            cmd += ' version {version}'.format(version=config_data.get('version'))
        if config_data.get('version_option'):
            cmd += ' {version}'.format(version=config_data.get('version_option'))
        if config_data.get('vrf'):
            cmd += ' vrf {vrf}'.format(vrf=config_data.get('vrf'))
        if config_data.get('community_string'):
            cmd += ' {community_string}'.format(community_string=config_data.get('community_string'))
        if config_data.get('traps'):
            for protocol in list(config_data.get('traps').keys()):
                cmd += ' {protocol}'.format(protocol=protocol)
    return cmd