from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def extender_modem_status(data, fos):
    vdom = data['vdom']
    extender_modem_status_data = data['extender_modem_status']
    filtered_data = underscore_to_hyphen(filter_extender_modem_status_data(extender_modem_status_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('extender', 'modem-status', data=converted_data, vdom=vdom)