from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.collections import is_string
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def perform_shutdown(self, task_vars, distribution):
    result = {}
    shutdown_result = {}
    shutdown_command_exec = self.get_shutdown_command(task_vars, distribution)
    self.cleanup(force=True)
    try:
        display.vvv('{action}: shutting down server...'.format(action=self._task.action))
        display.debug("{action}: shutting down server with command '{command}'".format(action=self._task.action, command=shutdown_command_exec))
        if self._play_context.check_mode:
            shutdown_result['rc'] = 0
        else:
            shutdown_result = self._low_level_execute_command(shutdown_command_exec, sudoable=self.DEFAULT_SUDOABLE)
    except AnsibleConnectionFailure as e:
        display.debug('{action}: AnsibleConnectionFailure caught and handled: {error}'.format(action=self._task.action, error=to_text(e)))
        shutdown_result['rc'] = 0
    if shutdown_result['rc'] != 0:
        result['failed'] = True
        result['shutdown'] = False
        result['msg'] = 'Shutdown command failed. Error was {stdout}, {stderr}'.format(stdout=to_native(shutdown_result['stdout'].strip()), stderr=to_native(shutdown_result['stderr'].strip()))
        return result
    result['failed'] = False
    result['shutdown_command'] = shutdown_command_exec
    return result