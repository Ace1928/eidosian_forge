from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def mkversion(major, minor, patch):
    return 1000 * 1000 * int(major) + 1000 * int(minor) + int(patch)