from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_vspan_session(self):
    """Builds up the session, adds the parameters that we specified, then creates it on the vSwitch"""
    session = vim.dvs.VmwareDistributedVirtualSwitch.VspanSession(name=self.name, enabled=True)
    if self.session_type is not None:
        session.sessionType = self.session_type
        if self.session_type == 'encapsulatedRemoteMirrorSource':
            self.check_source_port_received(session)
            self.check_source_port_transmitted(session)
            self.check_destination_port(session)
        if self.session_type == 'remoteMirrorSource':
            self.check_source_port_received(session)
            self.check_source_port_transmitted(session)
            self.check_destination_port(session)
        if self.session_type == 'remoteMirrorDest':
            self.check_source_port_received(session)
            self.check_destination_port(session)
        if self.session_type == 'dvPortMirror':
            self.check_source_port_received(session)
            self.check_source_port_transmitted(session)
            self.check_destination_port(session)
    self.check_self_properties(session)
    config_version = self.dv_switch.config.configVersion
    s_spec = vim.dvs.VmwareDistributedVirtualSwitch.VspanConfigSpec(vspanSession=session, operation='add')
    c_spec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec(vspanConfigSpec=[s_spec], configVersion=config_version)
    task = self.dv_switch.ReconfigureDvs_Task(c_spec)
    try:
        wait_for_task(task)
    except Exception:
        self.restore_original_state()
        self.module.fail_json(msg=task.info.error.msg)