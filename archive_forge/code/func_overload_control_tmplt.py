from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def overload_control_tmplt(config_data):
    config_data = config_data.get('overload_control', {})
    command = 'snmp-server overload-control'
    if config_data.get('overload_drop_time'):
        command += ' {overload_drop_time}'.format(overload_drop_time=config_data['overload_drop_time'])
    if config_data.get('overload_throttle_rate'):
        command += ' {overload_throttle_rate}'.format(overload_throttle_rate=config_data['overload_throttle_rate'])
    return command