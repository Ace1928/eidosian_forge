from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def get_shutdown_command(self, task_vars, distribution):
    reboot_command = self._task.args.get('reboot_command')
    if reboot_command is not None:
        try:
            reboot_command = check_type_str(reboot_command, allow_conversion=False)
        except TypeError as e:
            raise AnsibleError("Invalid value given for 'reboot_command': %s." % to_native(e))
        shutdown_bin = reboot_command.split(' ', 1)[0]
    else:
        shutdown_bin = self._get_value_from_facts('SHUTDOWN_COMMANDS', distribution, 'DEFAULT_SHUTDOWN_COMMAND')
    if shutdown_bin[0] == '/':
        return shutdown_bin
    else:
        default_search_paths = ['/sbin', '/bin', '/usr/sbin', '/usr/bin', '/usr/local/sbin']
        search_paths = self._task.args.get('search_paths', default_search_paths)
        try:
            search_paths = check_type_list(search_paths)
        except TypeError:
            err_msg = "'search_paths' must be a string or flat list of strings, got {0}"
            raise AnsibleError(err_msg.format(search_paths))
        display.debug('{action}: running find module looking in {paths} to get path for "{command}"'.format(action=self._task.action, command=shutdown_bin, paths=search_paths))
        find_result = self._execute_module(task_vars=task_vars, module_name='ansible.legacy.find', module_args={'paths': search_paths, 'patterns': [shutdown_bin], 'file_type': 'any'})
        full_path = [x['path'] for x in find_result['files']]
        if not full_path:
            raise AnsibleError('Unable to find command "{0}" in search paths: {1}'.format(shutdown_bin, search_paths))
        return full_path[0]