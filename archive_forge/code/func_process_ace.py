from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.argspec.acls.acls import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def process_ace(self, w_ace):
    command = ''
    ace_keys = w_ace.keys()
    if 'remark' in ace_keys:
        command += 'remark ' + w_ace['remark'] + ' '
    else:
        command += w_ace['grant'] + ' '
        if 'protocol' in ace_keys:
            if w_ace['protocol'] == 'icmpv6':
                command += 'icmp' + ' '
            else:
                command += w_ace['protocol'] + ' '
            src = self.get_address(w_ace['source'], w_ace['protocol'])
            dest = self.get_address(w_ace['destination'], w_ace['protocol'])
            command += src + dest
            if 'protocol_options' in ace_keys:
                pro = list(w_ace['protocol_options'].keys())[0]
                if pro != w_ace['protocol']:
                    self._module.fail_json(msg='protocol and protocol_options mismatch')
                flags = ''
                for k in w_ace['protocol_options'][pro].keys():
                    if k not in ['telemetry_queue', 'telemetry_path']:
                        k = re.sub('_', '-', k)
                    flags += k + ' '
                command += flags
            if 'dscp' in ace_keys:
                command += 'dscp ' + w_ace['dscp'] + ' '
            if 'fragments' in ace_keys:
                command += 'fragments '
            if 'precedence' in ace_keys:
                command += 'precedence ' + w_ace['precedence'] + ' '
        if 'log' in ace_keys:
            command += 'log '
    if 'sequence' in ace_keys:
        command = str(w_ace['sequence']) + ' ' + command
    return command