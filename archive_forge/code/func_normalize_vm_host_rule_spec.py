from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
def normalize_vm_host_rule_spec(self, rule_obj=None, cluster_obj=None):
    """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object
            cluster_obj: Cluster managed object

        Returns: Dictionary with DRS VM HOST Rule info

        """
    if not all([rule_obj, cluster_obj]):
        return {}
    return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vm_group_name=rule_obj.vmGroupName, rule_affine_host_group_name=rule_obj.affineHostGroupName, rule_anti_affine_host_group_name=rule_obj.antiAffineHostGroupName, rule_vms=self.get_all_from_group(group_name=rule_obj.vmGroupName, cluster_obj=cluster_obj), rule_affine_hosts=self.get_all_from_group(group_name=rule_obj.affineHostGroupName, cluster_obj=cluster_obj, hostgroup=True), rule_anti_affine_hosts=self.get_all_from_group(group_name=rule_obj.antiAffineHostGroupName, cluster_obj=cluster_obj, hostgroup=True), rule_type='vm_host_rule')