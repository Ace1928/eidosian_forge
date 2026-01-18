from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
def is_pyxcli_installed(module):
    if not PYXCLI_INSTALLED:
        module.fail_json(msg=missing_required_lib('pyxcli'), exception=PYXCLI_IMP_ERR)