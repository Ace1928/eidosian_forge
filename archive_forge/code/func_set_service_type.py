from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def set_service_type(self, vnic_manager, vmk, service_type, operation='select'):
    """
        Set service type to given VMKernel
        Args:
            vnic_manager: Virtual NIC manager object
            vmk: VMkernel managed object
            service_type: Name of service type
            operation: Select to select service type, deselect to deselect service type

        """
    try:
        if operation == 'select':
            if not self.module.check_mode:
                vnic_manager.SelectVnicForNicType(service_type, vmk.device)
        elif operation == 'deselect':
            if not self.module.check_mode:
                vnic_manager.DeselectVnicForNicType(service_type, vmk.device)
    except vmodl.fault.InvalidArgument as invalid_arg:
        self.module.fail_json(msg="Failed to %s VMK service type '%s' on '%s' due to : %s" % (operation, service_type, vmk.device, to_native(invalid_arg.msg)))