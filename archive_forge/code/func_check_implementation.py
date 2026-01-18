from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from os import path as os_path
import traceback
def check_implementation(conn, snote):
    check_implemented = call_rfc_method(conn, 'SCWB_API_GET_NOTES_IMPLEMENTED', {})
    for snote_list in check_implemented['ET_NOTES_IMPL']:
        if snote in snote_list['NUMM']:
            return True
    return False