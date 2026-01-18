from __future__ import (absolute_import, division, print_function)
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.task import Task
from ansible.module_utils.six import string_types
def notify_host(self, host):
    if not self.is_host_notified(host):
        self.notified_hosts.append(host)
        return True
    return False