from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def sctp_filter_profile(data, fos):
    vdom = data['vdom']
    state = data['state']
    sctp_filter_profile_data = data['sctp_filter_profile']
    filtered_data = underscore_to_hyphen(filter_sctp_filter_profile_data(sctp_filter_profile_data))
    if state == 'present' or state is True:
        return fos.set('sctp-filter', 'profile', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('sctp-filter', 'profile', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')