from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def assert_requirements_present(module):
    if DNSPYTHON_IMPORTERROR is not None:
        module.fail_json(msg=missing_required_lib('dnspython'), exception=DNSPYTHON_IMPORTERROR)