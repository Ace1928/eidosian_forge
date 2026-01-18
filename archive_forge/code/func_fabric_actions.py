from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def fabric_actions(rest_obj, module):
    """
    fabric management actions
    :param rest_obj: session object
    :param module: ansible module object
    :return: None
    """
    module_params = module.params
    state = module_params['state']
    name = module_params['name']
    all_fabrics = rest_obj.get_all_items_with_pagination(FABRIC_URI)['value']
    if state == 'present':
        create_modify_fabric(name, all_fabrics, rest_obj, module)
    else:
        delete_fabric(all_fabrics, rest_obj, module, name)