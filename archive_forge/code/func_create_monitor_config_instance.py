from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_monitor_config_instance(monitor_config):
    return MonitorConfig(profile_monitor_status=monitor_config['profile_monitor_status'], protocol=monitor_config['protocol'], port=monitor_config['port'], path=monitor_config['path'], interval_in_seconds=monitor_config['interval_in_seconds'], timeout_in_seconds=monitor_config['timeout_in_seconds'], tolerated_number_of_failures=monitor_config['tolerated_number_of_failures'])