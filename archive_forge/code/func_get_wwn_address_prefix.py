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
def get_wwn_address_prefix(starting_address):
    """Prefix wwnn and wwpn MAC address with 20x00 and 20x01 respectively"""
    delimiter, wwnn_prefix, wwpn_prefix = (None, None, None)
    if '.' in starting_address:
        delimiter = '.'
    elif ':' in starting_address:
        delimiter = ':'
    elif '-' in starting_address:
        delimiter = '-'
    length = len(starting_address.split(delimiter)[0])
    if length == 4:
        wwnn_prefix = '2000{0}'.format(delimiter)
        wwpn_prefix = '2001{0}'.format(delimiter)
    else:
        wwnn_prefix = '20{0}00{0}'.format(delimiter)
        wwpn_prefix = '20{0}01{0}'.format(delimiter)
    return (wwnn_prefix, wwpn_prefix)