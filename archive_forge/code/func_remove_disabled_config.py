from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def remove_disabled_config(self, diagnostic_setting):
    diagnostic_setting['logs'] = [log for log in diagnostic_setting.get('logs', []) if log.get('enabled')]
    diagnostic_setting['metrics'] = [metric for metric in diagnostic_setting.get('metrics', []) if metric.get('enabled')]
    return diagnostic_setting