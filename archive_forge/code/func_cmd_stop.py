from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.validation import check_type_int
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
def cmd_stop(self):
    result = dict()
    args_1 = self.get_up_cmd(self.check_mode, no_start=True)
    rc_1, stdout_1, stderr_1 = self.client.call_cli(*args_1, cwd=self.project_src)
    events_1 = self.parse_events(stderr_1, dry_run=self.check_mode)
    self.emit_warnings(events_1)
    self.update_result(result, events_1, stdout_1, stderr_1, ignore_service_pull_events=True)
    is_failed_1 = is_failed(events_1, rc_1)
    if not is_failed_1 and (not self._are_containers_stopped()):
        args_2 = self.get_stop_cmd(self.check_mode)
        rc_2, stdout_2, stderr_2 = self.client.call_cli(*args_2, cwd=self.project_src)
        events_2 = self.parse_events(stderr_2, dry_run=self.check_mode)
        self.emit_warnings(events_2)
        self.update_result(result, events_2, stdout_2, stderr_2)
    else:
        args_2 = []
        rc_2, stdout_2, stderr_2 = (0, b'', b'')
        events_2 = []
    self.update_failed(result, events_1 + events_2, args_1 if is_failed_1 else args_2, stdout_1 if is_failed_1 else stdout_2, stderr_1 if is_failed_1 else stderr_2, rc_1 if is_failed_1 else rc_2)
    return result