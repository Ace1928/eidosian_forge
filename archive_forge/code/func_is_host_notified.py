from __future__ import (absolute_import, division, print_function)
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.task import Task
from ansible.module_utils.six import string_types
def is_host_notified(self, host):
    return host in self.notified_hosts