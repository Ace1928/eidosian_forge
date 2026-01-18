from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def fail_on_error(self, error, api=None, stack=False, depth=1, previous_errors=None):
    """depth identifies how far is the caller in the call stack"""
    if error is None:
        return
    depth += 1
    if api is not None:
        error = 'calling api: %s: %s' % (api, error)
    results = dict(msg='Error in %s: %s' % (self.get_caller(depth), error))
    if stack:
        results['stack'] = traceback.format_stack()
    if previous_errors:
        results['previous_errors'] = ' - '.join(previous_errors)
    if getattr(self, 'ansible_module', None) is not None:
        self.ansible_module.fail_json(**results)
    raise AttributeError('Expecting self.ansible_module to be set when reporting %s' % repr(results))