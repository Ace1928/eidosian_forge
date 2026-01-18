from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def targets_tmplt(config_data):
    name = config_data.get('name', '')
    command = ''
    if name:
        command = 'snmp-server target list {name}'.format(name=name)
    if config_data.get('host'):
        command += ' host {host}'.format(host=config_data['host'])
    if config_data.get('vrf'):
        command += ' vrf {vrf}'.format(vrf=config_data['vrf'])
    return command