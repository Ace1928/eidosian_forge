from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import traceback
import json
def clean_output(o):
    return str(o)