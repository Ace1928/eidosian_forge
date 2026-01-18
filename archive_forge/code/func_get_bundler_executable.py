from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_bundler_executable(module):
    if module.params.get('executable'):
        result = module.params.get('executable').split(' ')
    else:
        result = [module.get_bin_path('bundle', True)]
    return result