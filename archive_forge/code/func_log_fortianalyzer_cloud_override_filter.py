from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def log_fortianalyzer_cloud_override_filter(data, fos):
    vdom = data['vdom']
    log_fortianalyzer_cloud_override_filter_data = data['log_fortianalyzer_cloud_override_filter']
    filtered_data = underscore_to_hyphen(filter_log_fortianalyzer_cloud_override_filter_data(log_fortianalyzer_cloud_override_filter_data))
    return fos.set('log.fortianalyzer-cloud', 'override-filter', data=filtered_data, vdom=vdom)