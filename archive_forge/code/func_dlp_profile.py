from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def dlp_profile(data, fos):
    vdom = data['vdom']
    state = data['state']
    dlp_profile_data = data['dlp_profile']
    dlp_profile_data = flatten_multilists_attributes(dlp_profile_data)
    filtered_data = underscore_to_hyphen(filter_dlp_profile_data(dlp_profile_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    if state == 'present' or state is True:
        return fos.set('dlp', 'profile', data=converted_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('dlp', 'profile', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')