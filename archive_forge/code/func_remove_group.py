from __future__ import (absolute_import, division, print_function)
import sys
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.inventory.group import Group
from ansible.inventory.host import Host
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.utils.path import basedir
def remove_group(self, group):
    if group in self.groups:
        del self.groups[group]
        display.debug('Removed group %s from inventory' % group)
        self._groups_dict_cache = {}
    for host in self.hosts:
        h = self.hosts[host]
        h.remove_group(group)