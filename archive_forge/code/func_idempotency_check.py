from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils.six import string_types
def idempotency_check(self, old_params, new_params):
    """
        Return True if something changed. Function will use fields from module_arg_spec to perform dependency checks.
        :param old_params: old parameters dictionary, body from Get request.
        :param new_params: new parameters dictionary, unpacked module parameters.
        """
    modifiers = {}
    result = {}
    self.create_compare_modifiers(self.module.argument_spec, '', modifiers)
    self.results['modifiers'] = modifiers
    return self.default_compare(modifiers, new_params, old_params, '', self.results)