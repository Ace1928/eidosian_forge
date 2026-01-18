from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def set_port_security_promiscuous(self, ports, state):
    """Set the given port to the given promiscuous state.
        Parameters
        ----------
        port : str[]
            PortKey
        state: bool
            State of the promiscuous mode, if true its allowed, else not.
        """
    port_spec = []
    port_policy = vim.dvs.VmwareDistributedVirtualSwitch.MacManagementPolicy(allowPromiscuous=state)
    port_settings = vim.dvs.VmwareDistributedVirtualSwitch.VmwarePortConfigPolicy(macManagementPolicy=port_policy)
    for port in ports:
        temp_port_spec = vim.dvs.DistributedVirtualPort.ConfigSpec(operation='edit', key=port, setting=port_settings)
        port_spec.append(temp_port_spec)
    task = self.dv_switch.ReconfigureDVPort_Task(port_spec)
    try:
        wait_for_task(task)
    except Exception:
        self.restore_original_state()
        self.module.fail_json(msg=task.info.error.msg)