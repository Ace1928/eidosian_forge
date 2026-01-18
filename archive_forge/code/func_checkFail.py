from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def checkFail():
    if failed:
        module.fail_json(msg=msg)
    else:
        return True