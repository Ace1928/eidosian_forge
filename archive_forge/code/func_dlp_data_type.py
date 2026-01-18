from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def dlp_data_type(data, fos):
    vdom = data['vdom']
    state = data['state']
    dlp_data_type_data = data['dlp_data_type']
    filtered_data = underscore_to_hyphen(filter_dlp_data_type_data(dlp_data_type_data))
    if state == 'present' or state is True:
        return fos.set('dlp', 'data-type', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('dlp', 'data-type', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')