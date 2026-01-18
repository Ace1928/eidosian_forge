from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_recovery_state(self):
    """Get if the service is in recovery mode."""
    self.pg_info['in_recovery'] = self.__exec_sql('SELECT pg_is_in_recovery()')[0]['pg_is_in_recovery']