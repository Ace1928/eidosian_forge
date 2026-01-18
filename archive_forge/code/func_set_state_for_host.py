from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def set_state_for_host(self, hostname: str, state: HostState) -> None:
    if not isinstance(state, HostState):
        raise AnsibleAssertionError('Expected state to be a HostState but was a %s' % type(state))
    self._host_states[hostname] = state