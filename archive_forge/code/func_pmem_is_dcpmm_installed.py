from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_is_dcpmm_installed(self):
    command = ['show', '-system', '-capabilities']
    return self.pmem_run_ipmctl(command)