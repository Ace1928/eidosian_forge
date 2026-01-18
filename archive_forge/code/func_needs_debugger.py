from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.parsing.dataloader import DataLoader
from ansible.vars.clean import module_response_deepcopy, strip_internal_keys
def needs_debugger(self, globally_enabled=False):
    _debugger = self._task_fields.get('debugger')
    _ignore_errors = C.TASK_DEBUGGER_IGNORE_ERRORS and self._task_fields.get('ignore_errors')
    ret = False
    if globally_enabled and (self.is_failed() and (not _ignore_errors) or self.is_unreachable()):
        ret = True
    if _debugger in ('always',):
        ret = True
    elif _debugger in ('never',):
        ret = False
    elif _debugger in ('on_failed',) and self.is_failed() and (not _ignore_errors):
        ret = True
    elif _debugger in ('on_unreachable',) and self.is_unreachable():
        ret = True
    elif _debugger in ('on_skipped',) and self.is_skipped():
        ret = True
    return ret