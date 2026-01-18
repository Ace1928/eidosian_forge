from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def mark_host_failed(self, host):
    s = self.get_host_state(host)
    display.debug('marking host %s failed, current state: %s' % (host, s))
    if s.run_state == IteratingStates.HANDLERS:
        s.run_state = s.pre_flushing_run_state
        s.update_handlers = True
    s = self._set_failed_state(s)
    display.debug('^ failed state is now: %s' % s)
    self.set_state_for_host(host.name, s)
    self._play._removed_hosts.append(host.name)