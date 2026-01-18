from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.validation import check_type_int
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
def cmd_down(self):
    result = dict()
    args = self.get_down_cmd(self.check_mode)
    rc, stdout, stderr = self.client.call_cli(*args, cwd=self.project_src)
    events = self.parse_events(stderr, dry_run=self.check_mode)
    self.emit_warnings(events)
    self.update_result(result, events, stdout, stderr)
    self.update_failed(result, events, args, stdout, stderr, rc)
    return result