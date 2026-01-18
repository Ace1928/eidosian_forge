from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def is_universal_timeout_enabled(payload):
    u_sess_timeout = -1
    for up in payload:
        if up.get('SessionType') == 'UniversalTimeout':
            u_sess_timeout = up.get('SessionTimeout')
            break
    return u_sess_timeout > 0