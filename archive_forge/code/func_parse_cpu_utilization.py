from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_cpu_utilization(self, data):
    facts = {}
    regex_cpu_utilization = re.compile('\n            (^Core\\s(?P<core>\\d+)?:)?\n            (^|\\s)CPU\\sutilization\\sfor\\sfive\\sseconds:\n            (\\s(?P<f_sec>\\d+)?%)?\n            (\\s(?P<f_se_nom>\\d+)%/(?P<f_s_denom>\\d+)%\\)?)?\n            ;\\sone\\sminute:\\s(?P<a_min>\\d+)?%\n            ;\\sfive\\sminutes:\\s(?P<f_min>\\d+)?%\n            ', re.VERBOSE)
    for line in data.split('\n'):
        match_cpu_utilization = regex_cpu_utilization.match(line)
        if match_cpu_utilization:
            _core = 'core'
            if match_cpu_utilization.group('core'):
                _core = 'core_' + str(match_cpu_utilization.group('core'))
            facts[_core] = {}
            facts[_core]['five_seconds'] = int(match_cpu_utilization.group('f_se_nom') or match_cpu_utilization.group('f_sec'))
            facts[_core]['one_minute'] = int(match_cpu_utilization.group('a_min'))
            facts[_core]['five_minutes'] = int(match_cpu_utilization.group('f_min'))
            if match_cpu_utilization.group('f_s_denom'):
                facts[_core]['five_seconds_interrupt'] = int(match_cpu_utilization.group('f_s_denom'))
    return facts