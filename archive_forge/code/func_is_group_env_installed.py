from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def is_group_env_installed(self, name):
    name_lower = name.lower()
    if yum.__version_info__ >= (3, 4):
        groups_list = self.yum_base.doGroupLists(return_evgrps=True)
    else:
        groups_list = self.yum_base.doGroupLists()
    groups = groups_list[0]
    for group in groups:
        if name_lower.endswith(group.name.lower()) or name_lower.endswith(group.groupid.lower()):
            return True
    if yum.__version_info__ >= (3, 4):
        envs = groups_list[2]
        for env in envs:
            if name_lower.endswith(env.name.lower()) or name_lower.endswith(env.environmentid.lower()):
                return True
    return False