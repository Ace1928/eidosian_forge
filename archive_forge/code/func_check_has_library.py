from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
import ssl
import traceback
def check_has_library(module):
    if not HAS_LIB:
        module.fail_json(msg=missing_required_lib('librouteros'), exception=LIB_IMP_ERR)