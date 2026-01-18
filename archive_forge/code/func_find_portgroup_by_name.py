from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def find_portgroup_by_name(self, host_system, portgroup_name, vswitch_name):
    """
        Find and return port group managed object
        Args:
            host_system: Name of Host System
            portgroup_name: Name of the Port Group
            vswitch_name: Name of the vSwitch

        Returns: Port Group managed object if found, else None
        """
    portgroups = self.get_all_port_groups_by_host(host_system=host_system)
    for portgroup in portgroups:
        if portgroup.spec.name == portgroup_name and portgroup.spec.vswitchName != vswitch_name:
            self.module.fail_json(msg="The portgroup already exists on vSwitch '%s'" % portgroup.spec.vswitchName)
        if portgroup.spec.name == portgroup_name and portgroup.spec.vswitchName == vswitch_name:
            return portgroup
    return None