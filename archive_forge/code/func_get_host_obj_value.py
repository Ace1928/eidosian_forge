from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_host_obj_value(self, host_dict):
    """
        Form host access value using host object
        :host_dict Host access type info
        :return Host object
        """
    if host_dict.get('host_id'):
        return self.get_host_obj(host_id=host_dict.get('host_id'))
    elif host_dict.get('host_name'):
        return self.get_host_obj(host_name=host_dict.get('host_name'))
    elif host_dict.get('ip_address'):
        return self.get_host_obj(ip_address=host_dict.get('ip_address'))