from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def append_result(self, result, failed=False):
    result_info = result._result
    task_info = result._task.serialize()
    task_info['args'] = None
    value = {}
    value['result'] = result_info
    value['task'] = task_info
    value['failed'] = failed
    if self.report_type == 'proxy':
        value = self.drop_nones(value)
    host = result._host.get_name()
    self.items[host].append(value)
    self.check_mode = result._task.check_mode
    if 'ansible_facts' in result_info:
        self.facts[host].update(result_info['ansible_facts'])