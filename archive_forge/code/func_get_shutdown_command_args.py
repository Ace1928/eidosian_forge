from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def get_shutdown_command_args(self, distribution):
    reboot_command = self._task.args.get('reboot_command')
    if reboot_command is not None:
        try:
            reboot_command = check_type_str(reboot_command, allow_conversion=False)
        except TypeError as e:
            raise AnsibleError("Invalid value given for 'reboot_command': %s." % to_native(e))
        try:
            return reboot_command.split(' ', 1)[1]
        except IndexError:
            return ''
    else:
        args = self._get_value_from_facts('SHUTDOWN_COMMAND_ARGS', distribution, 'DEFAULT_SHUTDOWN_COMMAND_ARGS')
        delay_min = self.pre_reboot_delay // 60
        reboot_message = self._task.args.get('msg', self.DEFAULT_REBOOT_MESSAGE)
        return args.format(delay_sec=self.pre_reboot_delay, delay_min=delay_min, message=reboot_message)