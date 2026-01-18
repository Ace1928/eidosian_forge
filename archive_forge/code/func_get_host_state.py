from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def get_host_state(self, host):
    if host.name not in self._host_states:
        self.set_state_for_host(host.name, HostState(blocks=[]))
    return self._host_states[host.name].copy()