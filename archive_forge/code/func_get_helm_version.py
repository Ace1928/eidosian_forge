from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
import re
import json
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
def get_helm_version(self):
    command = self.get_helm_binary() + ' version'
    rc, out, err = self.run_command(command)
    m = re.match('version.BuildInfo{Version:"v([0-9\\.]*)",', out)
    if m:
        return m.group(1)
    m = re.match('Client: &version.Version{SemVer:"v([0-9\\.]*)", ', out)
    if m:
        return m.group(1)
    return None