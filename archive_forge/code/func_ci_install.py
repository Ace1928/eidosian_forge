from __future__ import absolute_import, division, print_function
import json
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
def ci_install(self):
    return self._exec(['ci'])