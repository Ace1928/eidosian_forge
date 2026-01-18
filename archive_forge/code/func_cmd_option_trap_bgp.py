from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def cmd_option_trap_bgp(config_data):
    cmd = ''
    conf = config_data.get('traps', {}).get('bgp', {})
    if conf:
        if conf.get('enable'):
            cmd += 'snmp-server enable traps bgp'
        if conf.get('state_changes'):
            if conf.get('state_changes').get('enable'):
                cmd += ' state-changes'
            if conf.get('state_changes').get('all'):
                cmd += ' all'
            if conf.get('state_changes').get('backward_trans'):
                cmd += ' backward-trans'
            if conf.get('state_changes').get('limited'):
                cmd += ' limited'
        if conf.get('threshold'):
            cmd += ' threshold'
            if conf.get('threshold').get('prefix'):
                cmd += ' prefix'
    return cmd