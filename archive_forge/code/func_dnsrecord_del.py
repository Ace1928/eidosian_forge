from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def dnsrecord_del(self, zone_name=None, record_name=None, details=None):
    item = get_dnsrecord_dict(details)
    item.update(idnsname=record_name)
    return self._post_json(method='dnsrecord_del', name=zone_name, item=item)