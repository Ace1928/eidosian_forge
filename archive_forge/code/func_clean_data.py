from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def clean_data(data):
    """
    data: A dictionary.
    return: A data dictionary after removing items that are not required for end user.
    """
    for k in data.copy():
        if isinstance(data[k], dict):
            if data[k].get('@odata.id'):
                del data[k]['@odata.id']
        if not data[k]:
            del data[k]
    return data