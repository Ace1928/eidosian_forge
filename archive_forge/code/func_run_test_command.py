from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def run_test_command(self, distribution, **kwargs):
    test_command = self._task.args.get('test_command', self._get_value_from_facts('TEST_COMMANDS', distribution, 'DEFAULT_TEST_COMMAND'))
    display.vvv('{action}: attempting post-reboot test command'.format(action=self._task.action))
    display.debug("{action}: attempting post-reboot test command '{command}'".format(action=self._task.action, command=test_command))
    try:
        command_result = self._low_level_execute_command(test_command, sudoable=self.DEFAULT_SUDOABLE)
    except Exception:
        try:
            self._connection.reset()
        except AttributeError:
            pass
        raise
    if command_result['rc'] != 0:
        msg = 'Test command failed: {err} {out}'.format(err=to_native(command_result['stderr']), out=to_native(command_result['stdout']))
        raise RuntimeError(msg)
    display.vvv('{action}: system successfully rebooted'.format(action=self._task.action))