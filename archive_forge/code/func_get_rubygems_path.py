from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_rubygems_path(module):
    if module.params['executable']:
        result = module.params['executable'].split(' ')
    else:
        result = [module.get_bin_path('gem', True)]
    return result