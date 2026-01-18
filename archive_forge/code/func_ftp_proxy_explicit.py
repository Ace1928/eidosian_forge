from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def ftp_proxy_explicit(data, fos):
    vdom = data['vdom']
    ftp_proxy_explicit_data = data['ftp_proxy_explicit']
    ftp_proxy_explicit_data = flatten_multilists_attributes(ftp_proxy_explicit_data)
    filtered_data = underscore_to_hyphen(filter_ftp_proxy_explicit_data(ftp_proxy_explicit_data))
    return fos.set('ftp-proxy', 'explicit', data=filtered_data, vdom=vdom)