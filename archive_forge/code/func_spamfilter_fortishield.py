from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def spamfilter_fortishield(data, fos):
    vdom = data['vdom']
    spamfilter_fortishield_data = data['spamfilter_fortishield']
    filtered_data = underscore_to_hyphen(filter_spamfilter_fortishield_data(spamfilter_fortishield_data))
    return fos.set('spamfilter', 'fortishield', data=filtered_data, vdom=vdom)