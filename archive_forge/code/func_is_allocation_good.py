from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def is_allocation_good(self, ipmctl_out, command):
    warning = re.compile('WARNING')
    error = re.compile('.*Error.*')
    ignore_error = re.compile('Do you want to continue? [y/n] Error: Invalid data input.')
    errmsg = ''
    rc = True
    for line in ipmctl_out.splitlines():
        if warning.match(line):
            errmsg = '%s (command: %s)' % (line, command)
            rc = False
            break
        elif error.match(line):
            if not ignore_error:
                errmsg = '%s (command: %s)' % (line, command)
                rc = False
                break
    return (rc, errmsg)