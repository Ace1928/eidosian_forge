from __future__ import (absolute_import, division, print_function)
import abc
import re
from os.path import basename
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_pids_by_name(self, name):
    return [p.pid for p in self._process_iter(*self.NAME_ATTRS) if self._has_name(p, name)]