from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def delete_directory_user(rest_obj, user_id):
    delete_uri, changed = ("{0}('{1}')".format(ACCOUNT_URI, user_id), False)
    msg = 'Invalid domain user group name provided.'
    resp = rest_obj.invoke_request('DELETE', delete_uri)
    if resp.status_code == 204:
        changed = True
        msg = 'Successfully deleted the domain user group.'
    return (msg, changed)