from __future__ import (absolute_import, division, print_function)
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.task import Task
from ansible.module_utils.six import string_types
def clear_hosts(self):
    self.notified_hosts = []