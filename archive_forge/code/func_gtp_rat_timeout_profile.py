from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def gtp_rat_timeout_profile(data, fos):
    vdom = data['vdom']
    state = data['state']
    gtp_rat_timeout_profile_data = data['gtp_rat_timeout_profile']
    filtered_data = underscore_to_hyphen(filter_gtp_rat_timeout_profile_data(gtp_rat_timeout_profile_data))
    if state == 'present' or state is True:
        return fos.set('gtp', 'rat-timeout-profile', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('gtp', 'rat-timeout-profile', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')