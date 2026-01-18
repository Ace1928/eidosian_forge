from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def get_all_nw_sid():
    nw_sid = list()
    if os.path.isdir('/sapmnt'):
        for sid in os.listdir('/sapmnt'):
            if os.path.isdir('/usr/sap/' + sid):
                nw_sid = nw_sid + [sid]
            elif os.path.isdir('/sapmnt/' + sid + '/sap_bobj'):
                nw_sid = nw_sid + [sid]
    if nw_sid:
        return nw_sid