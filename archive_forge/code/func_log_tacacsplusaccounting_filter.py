from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def log_tacacsplusaccounting_filter(data, fos):
    vdom = data['vdom']
    log_tacacsplusaccounting_filter_data = data['log_tacacsplusaccounting_filter']
    filtered_data = underscore_to_hyphen(filter_log_tacacsplusaccounting_filter_data(log_tacacsplusaccounting_filter_data))
    return fos.set('log.tacacs+accounting', 'filter', data=filtered_data, vdom=vdom)