from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
def gather_drs_rule_info(self):
    """
        Gather DRS rule information about given cluster
        Returns: Dictionary of clusters with DRS information

        """
    cluster_rule_info = dict()
    for cluster_obj in self.cluster_obj_list:
        cluster_rule_info[cluster_obj.name] = []
        for drs_rule in cluster_obj.configuration.rule:
            if isinstance(drs_rule, vim.cluster.VmHostRuleInfo):
                cluster_rule_info[cluster_obj.name].append(self.normalize_vm_host_rule_spec(rule_obj=drs_rule, cluster_obj=cluster_obj))
            else:
                cluster_rule_info[cluster_obj.name].append(self.normalize_vm_vm_rule_spec(rule_obj=drs_rule))
    return cluster_rule_info