from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def casb_profile(data, fos):
    vdom = data['vdom']
    state = data['state']
    casb_profile_data = data['casb_profile']
    casb_profile_data = flatten_multilists_attributes(casb_profile_data)
    filtered_data = underscore_to_hyphen(filter_casb_profile_data(casb_profile_data))
    if state == 'present' or state is True:
        return fos.set('casb', 'profile', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('casb', 'profile', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')