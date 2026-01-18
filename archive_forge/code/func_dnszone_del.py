from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def dnszone_del(self, zone_name=None, record_name=None, details=None):
    return self._post_json(method='dnszone_del', name=zone_name, item={})