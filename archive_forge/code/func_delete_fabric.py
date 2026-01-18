from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def delete_fabric(all_fabrics, rest_obj, module, name):
    """
    deletes the fabric specified
    :param all_fabrics: All available fabric in system
    :param rest_obj: session object
    :param module: ansible module object
    :param name: fabric name specified
    :return: None
    """
    fabric_id = check_fabric_exits_for_state_absent(all_fabrics, module, name)
    rest_obj.invoke_request('DELETE', FABRIC_ID_URI.format(fabric_id=fabric_id))
    module.exit_json(msg='Fabric deletion operation is initiated.', fabric_id=fabric_id, changed=True)