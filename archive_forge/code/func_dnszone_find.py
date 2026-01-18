from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def dnszone_find(self, zone_name, details=None):
    items = {'all': 'true', 'idnsname': zone_name}
    if details is not None:
        items.update(details)
    return self._post_json(method='dnszone_find', name=zone_name, item=items)