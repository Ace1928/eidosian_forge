from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
def get_all_from_group(self, group_name=None, cluster_obj=None, hostgroup=False):
    """
        Return all VM / Host names using given group name
        Args:
            group_name: Rule name
            cluster_obj: Cluster managed object
            hostgroup: True if we want only host name from group

        Returns: List of VM / Host names belonging to given group object

        """
    obj_name_list = []
    if not all([group_name, cluster_obj]):
        return obj_name_list
    for group in cluster_obj.configurationEx.group:
        if group.name == group_name:
            if not hostgroup and isinstance(group, vim.cluster.VmGroup):
                obj_name_list = [vm.name for vm in group.vm]
                break
            if hostgroup and isinstance(group, vim.cluster.HostGroup):
                obj_name_list = [host.name for host in group.host]
                break
    return obj_name_list