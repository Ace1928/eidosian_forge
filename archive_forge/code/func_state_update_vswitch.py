from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def state_update_vswitch(self):
    """
        Update vSwitch

        """
    changed = False
    results = dict(changed=False, result="No change in vSwitch '%s'" % self.switch)
    spec = self.vss.spec
    if self.vss.mtu != self.mtu:
        spec.mtu = self.mtu
        changed = True
    if spec.numPorts != self.number_of_ports:
        spec.numPorts = self.number_of_ports
        changed = True
    nics_current = set(map(lambda n: n.rsplit('-', 1)[1], self.vss.pnic))
    if nics_current != set(self.nics):
        if self.nics:
            spec.bridge = vim.host.VirtualSwitch.BondBridge(nicDevice=self.nics)
        else:
            spec.bridge = None
        changed = True
        if not self.params['teaming']:
            nicOrder = spec.policy.nicTeaming.nicOrder
            if nicOrder.activeNic != [i for i in nicOrder.activeNic if i in self.nics]:
                nicOrder.activeNic = [i for i in nicOrder.activeNic if i in self.nics]
            if nicOrder.standbyNic != [i for i in nicOrder.standbyNic if i in self.nics]:
                nicOrder.standbyNic = [i for i in nicOrder.standbyNic if i in self.nics]
            if set(self.nics) - nics_current:
                nicOrder.activeNic += set(self.nics) - nics_current
    if self.update_security_policy(spec, results):
        changed = True
    if self.update_teaming_policy(spec, results):
        changed = True
    if self.update_traffic_shaping_policy(spec, results):
        changed = True
    if changed:
        if self.module.check_mode:
            results['msg'] = "vSwitch '%s' would be updated" % self.switch
        else:
            try:
                self.network_mgr.UpdateVirtualSwitch(vswitchName=self.switch, spec=spec)
                results['result'] = "vSwitch '%s' is updated successfully" % self.switch
            except vim.fault.ResourceInUse as resource_used:
                self.module.fail_json(msg="Failed to update vSwitch '%s' as physical network adapter being bridged is already in use: %s" % (self.switch, to_native(resource_used.msg)))
            except vim.fault.NotFound as not_found:
                self.module.fail_json(msg="Failed to update vSwitch with name '%s' as it does not exists: %s" % (self.switch, to_native(not_found.msg)))
            except vim.fault.HostConfigFault as host_config_fault:
                self.module.fail_json(msg="Failed to update vSwitch '%s' due to host configuration fault : %s" % (self.switch, to_native(host_config_fault.msg)))
            except vmodl.fault.InvalidArgument as invalid_argument:
                self.module.fail_json(msg="Failed to update vSwitch '%s', this can be due to either of following : 1. vSwitch Name exceeds the maximum allowed length, 2. Number of ports specified falls out of valid range, 3. Network policy is invalid, 4. Beacon configuration is invalid : %s" % (self.switch, to_native(invalid_argument.msg)))
            except vmodl.fault.SystemError as system_error:
                self.module.fail_json(msg="Failed to update vSwitch '%s' due to : %s" % (self.switch, to_native(system_error.msg)))
            except vmodl.fault.NotSupported as not_supported:
                self.module.fail_json(msg="Failed to update vSwitch '%s' as network adapter teaming policy is set but is not supported : %s" % (self.switch, to_native(not_supported.msg)))
            except Exception as generic_exc:
                self.module.fail_json(msg="Failed to update vSwitch '%s' due to generic exception : %s" % (self.switch, to_native(generic_exc)))
        results['changed'] = True
    self.module.exit_json(**results)