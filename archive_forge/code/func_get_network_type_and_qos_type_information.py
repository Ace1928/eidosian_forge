from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_network_type_and_qos_type_information(rest_obj):
    """
    rest_obj: Object containing information about connection to device.
    return: Dictionary with information for "Type" and "QosType" keys.
    """
    network_type_dict = get_type_information(rest_obj, NETWORK_TYPE_BASE_URI)
    qos_type_dict = get_type_information(rest_obj, QOS_TYPE_BASE_URI)
    for key, item in network_type_dict.items():
        item['QosType'] = qos_type_dict[item['QosType']]
    return network_type_dict