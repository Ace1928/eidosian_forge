from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def ssl_policy_name(self, policy_name):
    if policy_name == 'AppGwSslPolicy20150501':
        return 'ssl_policy20150501'
    elif policy_name == 'AppGwSslPolicy20170401':
        return 'ssl_policy20170401'
    elif policy_name == 'AppGwSslPolicy20170401S':
        return 'ssl_policy20170401_s'
    return None