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
def mac_validation(mac_input):
    match_found = re.match('[0-9a-f]{2}([-:])[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$|([0-9a-f]{4}([.])[0-9a-f]{4}([.])[0-9a-f]{4})$', mac_input.lower())
    return match_found