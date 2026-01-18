from __future__ import absolute_import, division, print_function
import binascii
from collections import defaultdict
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def decode_hex(hexstring):
    if len(hexstring) < 3:
        return hexstring
    if hexstring[:2] == '0x':
        return to_text(binascii.unhexlify(hexstring[2:]))
    return hexstring