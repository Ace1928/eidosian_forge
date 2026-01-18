from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_wanopt_webcache_data(json):
    option_list = ['always_revalidate', 'cache_by_default', 'cache_cookie', 'cache_expired', 'default_ttl', 'external', 'fresh_factor', 'host_validate', 'ignore_conditional', 'ignore_ie_reload', 'ignore_ims', 'ignore_pnc', 'max_object_size', 'max_ttl', 'min_ttl', 'neg_resp_time', 'reval_pnc']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary