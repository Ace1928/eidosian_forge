from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def delete_replaced_dict_config(**in_args):
    """ Create and enqueue deletion requests for the appropriate attributes in the dictionary
        specified by "dict_key". Update the input deletion_dict with the deleted attributes.
        The input 'inargs' is assumed to contain the following keyword arguments:

        cfg_key_set: The set of currently configured keys for the target dict

        cmd_key_set: The set of currently requested update keys for the target dict

        cfg_parent_dict: The configured dictionary containing the input key set

        uri_attr: a dictionary specifying REST URIs keyed by argspec keys

        uri_dict_key: The key for top level attribue to be used for uri lookup. If set
        to the string value 'cfg_dict_member_key', the current value of 'cfg_dict_member_key'
        is used. Otherwise, the specified value is used directly.

        deletion_dict: a dictionary containing attributes deleted from the parent dict

        requests: The list of REST API requests for the executing playbook section
        """
    uri_key = in_args['uri_dict_key']
    for cfg_dict_member_key in in_args['cfg_key_set'].difference(in_args['cmd_key_set']):
        cfg_dict_member_val = in_args['cfg_parent_dict'][cfg_dict_member_key]
        if in_args['uri_dict_key'] == 'cfg_dict_member_key':
            uri_key = cfg_dict_member_key
        uri = in_args['uri_attr'][uri_key]
        in_args['deletion_dict'].update({cfg_dict_member_key: cfg_dict_member_val})
        if isinstance(uri, dict):
            for member_key in uri:
                if in_args['cfg_parent_dict'].get(member_key) is not None:
                    request = {'path': uri[member_key], 'method': DELETE}
                    in_args['requests'].append(request)
        elif isinstance(uri, list):
            for set_uri_item in uri:
                request = {'path': set_uri_item, 'method': DELETE}
        else:
            request = {'path': uri, 'method': DELETE}
            in_args['requests'].append(request)