from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
def gather_info(self):
    """
        Gather DRS group information about given cluster
        Returns: Dictionary of clusters with DRS groups

        """
    cluster_group_info = dict()
    for cluster_obj in self.cluster_obj_list:
        cluster_group_info[cluster_obj.name] = []
        for drs_group in cluster_obj.configurationEx.group:
            cluster_group_info[cluster_obj.name].append(self.__normalize_group_data(drs_group))
    self.__set_result(cluster_group_info)