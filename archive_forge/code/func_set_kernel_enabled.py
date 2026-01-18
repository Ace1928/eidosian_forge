from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_lines
def set_kernel_enabled(module, grubby_bin, value):
    rc, stdout, stderr = module.run_command([grubby_bin, '--update-kernel=ALL', '--remove-args' if value else '--args', 'selinux=0'])
    if rc != 0:
        if value:
            module.fail_json(msg='unable to remove selinux=0 from kernel config')
        else:
            module.fail_json(msg='unable to add selinux=0 to kernel config')