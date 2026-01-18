from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def query_service_type_for_vmks(self, service_type):
    """
        Return list of VMKernels
        Args:
            service_type: Name of service type

        Returns: List of VMKernel which belongs to that service type

        """
    vmks_list = []
    query = None
    try:
        query = self.esxi_host_obj.configManager.virtualNicManager.QueryNetConfig(service_type)
    except vim.fault.HostConfigFault as config_fault:
        self.module.fail_json(msg='Failed to get all VMKs for service type %s due to host config fault : %s' % (service_type, to_native(config_fault.msg)))
    except vmodl.fault.InvalidArgument as invalid_argument:
        self.module.fail_json(msg='Failed to get all VMKs for service type %s due to invalid arguments : %s' % (service_type, to_native(invalid_argument.msg)))
    if not query.selectedVnic:
        return vmks_list
    vnics_with_service_type = [vnic.device for vnic in query.candidateVnic if vnic.key in query.selectedVnic]
    return vnics_with_service_type