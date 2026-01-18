from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def is_any_block_rescuing(self, state):
    """
        Given the current HostState state, determines if the current block, or any child blocks,
        are in rescue mode.
        """
    if state.run_state == IteratingStates.TASKS and state.get_current_block().rescue:
        return True
    if state.tasks_child_state is not None:
        return self.is_any_block_rescuing(state.tasks_child_state)
    if state.rescue_child_state is not None:
        return self.is_any_block_rescuing(state.rescue_child_state)
    if state.always_child_state is not None:
        return self.is_any_block_rescuing(state.always_child_state)
    return False