from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
@staticmethod
def extract_node_info(node_list):
    summary = list()
    for node in node_list:
        node_dict = dict()
        for key, value in vars(node).items():
            if key in ['assigned_node_id', 'cip', 'mip', 'name', 'node_id', 'pending_node_id', 'sip']:
                node_dict[key] = value
        summary.append(node_dict)
    return summary