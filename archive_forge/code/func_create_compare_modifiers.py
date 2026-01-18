from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils.six import string_types
def create_compare_modifiers(self, arg_spec, path, result):
    for k in arg_spec.keys():
        o = arg_spec[k]
        updatable = o.get('updatable', True)
        comparison = o.get('comparison', 'default')
        disposition = o.get('disposition', '*')
        if disposition == '/':
            disposition = '/*'
        p = path + ('/' if len(path) > 0 else '') + disposition.replace('*', k) + ('/*' if o['type'] == 'list' else '')
        if comparison != 'default' or not updatable:
            result[p] = {'updatable': updatable, 'comparison': comparison}
        if o.get('options'):
            self.create_compare_modifiers(o.get('options'), p, result)