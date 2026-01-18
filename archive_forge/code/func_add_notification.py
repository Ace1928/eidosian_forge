from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def add_notification(self, hostname: str, notification: str) -> None:
    host_state = self._host_states[hostname]
    if notification not in host_state.handler_notifications:
        host_state.handler_notifications.append(notification)