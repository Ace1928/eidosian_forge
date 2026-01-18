from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def dnszone_add(self, zone_name=None, details=None):
    items = {}
    if details is not None:
        items.update(details)
    return self._post_json(method='dnszone_add', name=zone_name, item=items)