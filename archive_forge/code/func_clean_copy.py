from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.parsing.dataloader import DataLoader
from ansible.vars.clean import module_response_deepcopy, strip_internal_keys
def clean_copy(self):
    """ returns 'clean' taskresult object """
    result = TaskResult(self._host, self._task, {}, self._task_fields)
    if result._task and result._task.action in C._ACTION_DEBUG:
        ignore = _IGNORE + ('invocation',)
    else:
        ignore = _IGNORE
    subset = {}
    for sub in _SUB_PRESERVE:
        if sub in self._result:
            subset[sub] = {}
            for key in _SUB_PRESERVE[sub]:
                if key in self._result[sub]:
                    subset[sub][key] = self._result[sub][key]
    if isinstance(self._task.no_log, bool) and self._task.no_log or self._result.get('_ansible_no_log', False):
        x = {'censored': "the output has been hidden due to the fact that 'no_log: true' was specified for this result"}
        for preserve in _PRESERVE:
            if preserve in self._result:
                x[preserve] = self._result[preserve]
        result._result = x
    elif self._result:
        result._result = module_response_deepcopy(self._result)
        for remove_key in ignore:
            if remove_key in result._result:
                del result._result[remove_key]
        strip_internal_keys(result._result, exceptions=CLEAN_EXCEPTIONS)
    result._result.update(subset)
    return result