from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def get_all_hana_sid():
    hana_sid = list()
    if os.path.isdir('/hana/shared'):
        for sid in os.listdir('/hana/shared'):
            if os.path.isdir('/usr/sap/' + sid):
                hana_sid = hana_sid + [sid]
    if hana_sid:
        return hana_sid