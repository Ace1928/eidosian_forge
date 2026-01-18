from __future__ import (absolute_import, division, print_function)
import time
from ansible.errors import AnsibleError
from ansible.plugins.action import ActionBase
from ansible.playbook.task import Task
from ansible.utils.display import Display
def clean_async_result(reference_keys, obj):
    for key in reference_keys:
        obj.pop(key)
    return obj