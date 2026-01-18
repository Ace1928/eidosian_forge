from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def cmd_option_engine_id(config_data):
    cmd = ''
    if config_data:
        cmd = 'snmp-server engineID '
        if config_data.get('local'):
            cmd += 'local'
        if config_data.get('remote'):
            rm = config_data.get('remote')
            if rm.get('host'):
                cmd += 'remote {host}'.format(host=rm.get('host'))
            if rm.get('udp_port'):
                cmd += ' udp-port {udp_port}'.format(udp_port=rm.get('udp_port'))
            if rm.get('vrf'):
                cmd += ' vrf {vrf}'.format(vrf=rm.get('vrf'))
        if config_data.get('id'):
            cmd += ' {id}'.format(id=config_data.get('id'))
    return cmd