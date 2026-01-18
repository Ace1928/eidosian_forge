from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_host_parameters():
    """This method provides parameters required for the ansible host
    module on Unity"""
    return dict(host_name=dict(required=False, type='str'), host_id=dict(required=False, type='str'), description=dict(required=False, type='str'), host_os=dict(required=False, type='str', choices=['AIX', 'Citrix XenServer', 'HP-UX', 'IBM VIOS', 'Linux', 'Mac OS', 'Solaris', 'VMware ESXi', 'Windows Client', 'Windows Server']), new_host_name=dict(required=False, type='str'), initiators=dict(required=False, type='list', elements='str'), initiator_state=dict(required=False, type='str', choices=['present-in-host', 'absent-in-host']), network_address=dict(required=False, type='str'), network_address_state=dict(required=False, type='str', choices=['present-in-host', 'absent-in-host']), state=dict(required=True, type='str', choices=['present', 'absent']))