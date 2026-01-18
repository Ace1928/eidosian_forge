from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_macaddress(self, entryid):
    xml = etree.fromstring(self.find_entry(entryid).XMLDesc(0))
    try:
        result = xml.xpath('/network/mac')[0].get('address')
    except Exception:
        raise ValueError('MAC address not specified')
    return result