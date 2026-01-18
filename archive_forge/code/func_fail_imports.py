from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
import traceback
def fail_imports(module, needs_certifi=True):
    errors = []
    traceback = []
    if not HAS_REDIS_PACKAGE:
        errors.append(missing_required_lib('redis'))
        traceback.append(REDIS_IMP_ERR)
    if not HAS_CERTIFI_PACKAGE and needs_certifi:
        errors.append(missing_required_lib('certifi'))
        traceback.append(CERTIFI_IMPORT_ERROR)
    if errors:
        module.fail_json(msg='\n'.join(errors), traceback='\n'.join(traceback))