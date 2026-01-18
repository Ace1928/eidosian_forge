from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def parted_version():
    """
    Returns the major and minor version of parted installed on the system.
    """
    global module, parted_exec
    rc, out, err = module.run_command('%s --version' % parted_exec)
    if rc != 0:
        module.fail_json(msg='Failed to get parted version.', rc=rc, out=out, err=err)
    major, minor, rev = parse_parted_version(out)
    if major is None:
        module.fail_json(msg='Failed to get parted version.', rc=0, out=out)
    return (major, minor, rev)