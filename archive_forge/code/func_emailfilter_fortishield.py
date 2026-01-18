from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def emailfilter_fortishield(data, fos):
    vdom = data['vdom']
    emailfilter_fortishield_data = data['emailfilter_fortishield']
    filtered_data = underscore_to_hyphen(filter_emailfilter_fortishield_data(emailfilter_fortishield_data))
    return fos.set('emailfilter', 'fortishield', data=filtered_data, vdom=vdom)