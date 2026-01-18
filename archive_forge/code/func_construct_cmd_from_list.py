from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def construct_cmd_from_list(cmd, config):
    temp = []
    for k, v in iteritems(config):
        temp.append(v)
    cmd += ' ' + ' '.join(sorted(temp))
    return cmd