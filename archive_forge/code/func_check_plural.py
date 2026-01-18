from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def check_plural(src, dest):
    if isinstance(rule.get(src), list):
        rule[dest] = rule[src]
        rule[src] = None