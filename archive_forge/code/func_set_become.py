from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from ansible.plugins import AnsiblePlugin
def set_become(self, become_context):
    self._become = become_context.become
    self._become_pass = getattr(become_context, 'become_pass') or ''