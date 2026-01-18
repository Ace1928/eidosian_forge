from __future__ import (absolute_import, division, print_function)
import re
import json
import codecs
import binascii
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def mac_to_base64_conversion(mac_address, module):
    try:
        if mac_address:
            allowed_mac_separators = [':', '-', '.']
            for sep in allowed_mac_separators:
                if sep in mac_address:
                    b64_mac_address = codecs.encode(codecs.decode(mac_address.replace(sep, ''), 'hex'), 'base64')
                    address = codecs.decode(b64_mac_address, 'utf-8').rstrip()
                    return address
    except binascii.Error:
        module.fail_json(msg='Encoding of MAC address {0} to base64 failed'.format(mac_address))