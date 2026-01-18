from __future__ import absolute_import, division, print_function
import os
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import (
def find_template_kind(module, module_params):
    if 'kind' not in module_params:
        return module_params
    module_params['snippet'] = module_params['kind'] == 'snippet'
    if module_params['snippet']:
        module_params.pop('kind')
    else:
        module_params['kind'] = module.find_resource_by_name('template_kinds', module_params['kind'], thin=True)
    return module_params