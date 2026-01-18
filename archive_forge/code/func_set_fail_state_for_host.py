from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def set_fail_state_for_host(self, hostname: str, fail_state: FailedStates) -> None:
    if not isinstance(fail_state, FailedStates):
        raise AnsibleAssertionError('Expected fail_state to be a FailedStates but was %s' % type(fail_state))
    self._host_states[hostname].fail_state = fail_state