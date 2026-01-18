from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_elastigroup(module, is_update):
    do_not_update = module.params['do_not_update']
    name = module.params.get('name')
    eg = spotinst.aws_elastigroup.Elastigroup()
    description = module.params.get('description')
    if name is not None:
        eg.name = name
    if description is not None:
        eg.description = description
    expand_capacity(eg, module, is_update, do_not_update)
    expand_strategy(eg, module)
    expand_scaling(eg, module)
    expand_integrations(eg, module)
    expand_compute(eg, module, is_update, do_not_update)
    expand_multai(eg, module)
    expand_scheduled_tasks(eg, module)
    return eg