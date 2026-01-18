from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_multai(eg, module):
    multai_load_balancers = module.params.get('multai_load_balancers')
    eg_multai = expand_fields(multai_fields, module.params, 'Multai')
    if multai_load_balancers is not None:
        eg_multai_load_balancers = expand_list(multai_load_balancers, multai_lb_fields, 'MultaiLoadBalancer')
        if len(eg_multai_load_balancers) > 0:
            eg_multai.balancers = eg_multai_load_balancers
            eg.multai = eg_multai