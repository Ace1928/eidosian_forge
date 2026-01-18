from __future__ import absolute_import, division, print_function
from functools import total_ordering
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def pad_commands(commands, interface):
    commands.insert(0, 'interface {0}'.format(interface))