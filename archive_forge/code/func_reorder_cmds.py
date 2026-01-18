from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def reorder_cmds(cmds):
    """
    There is a bug in some image versions where bfd echo-interface and
    bfd echo-rx-interval need to be applied last for them to nvgen properly.
    """
    regex1 = re.compile('^bfd echo-interface')
    regex2 = re.compile('^bfd echo-rx-interval')
    filtered_cmds = [i for i in cmds if not regex1.match(i)]
    filtered_cmds = [i for i in filtered_cmds if not regex2.match(i)]
    echo_int_cmd = [i for i in cmds if regex1.match(i)]
    echo_rx_cmd = [i for i in cmds if regex2.match(i)]
    filtered_cmds.extend(echo_int_cmd)
    filtered_cmds.extend(echo_rx_cmd)
    return filtered_cmds