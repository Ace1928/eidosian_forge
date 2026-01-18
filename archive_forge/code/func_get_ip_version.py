from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
from ipaddress import ip_network
def get_ip_version(val):
    """Returns IP address version
        :param: val: IP address to be validated for version.
    """
    try:
        val = u'{0}'.format(val)
        ip = ip_network(val, strict=False)
        return ip.version
    except ValueError:
        return 0