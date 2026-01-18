from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def check_and_set_parameters(self, module):
    self.parameters = {}
    check_for_none = netapp_utils.has_feature(module, 'check_required_params_for_none')
    if check_for_none:
        required_keys = [key for key, value in module.argument_spec.items() if value.get('required')]
    for param in module.params:
        if module.params[param] is not None:
            self.parameters[param] = module.params[param]
        elif check_for_none and param in required_keys:
            module.fail_json(msg='%s requires a value, got: None' % param)
    return self.parameters