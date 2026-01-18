from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def metric_config_to_dict(self, metric_config):
    return dict(category=metric_config.get('category'), enabled=metric_config.get('enabled'), retention_policy=self.retention_policy_to_dict(metric_config.get('retention_policy')))