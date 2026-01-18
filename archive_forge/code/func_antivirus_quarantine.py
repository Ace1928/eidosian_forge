from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def antivirus_quarantine(data, fos):
    vdom = data['vdom']
    antivirus_quarantine_data = data['antivirus_quarantine']
    antivirus_quarantine_data = flatten_multilists_attributes(antivirus_quarantine_data)
    filtered_data = underscore_to_hyphen(filter_antivirus_quarantine_data(antivirus_quarantine_data))
    return fos.set('antivirus', 'quarantine', data=filtered_data, vdom=vdom)