from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
from ansible.module_utils._text import to_native
def check_if_options_are_valid(self, target):
    """Check if options are valid"""
    dest_hostname = target.get('hostname', None)
    if dest_hostname is None:
        self.module.fail_json(msg="Please specify hostname for the trap target as it's a required parameter")
    dest_port = target.get('port', None)
    if dest_port is None:
        self.module.fail_json(msg="Please specify port for the trap target as it's a required parameter")
    dest_community = target.get('community', None)
    if dest_community is None:
        self.module.fail_json(msg="Please specify community for the trap target as it's a required parameter")
    return (dest_hostname, dest_port, dest_community)