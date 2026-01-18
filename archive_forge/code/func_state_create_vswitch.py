from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def state_create_vswitch(self):
    """
        Create a virtual switch

        Source from
        https://github.com/rreubenur/pyvmomi-community-samples/blob/patch-1/samples/create_vswitch.py

        """
    results = dict(changed=False, result='')
    vss_spec = vim.host.VirtualSwitch.Specification()
    vss_spec.numPorts = self.number_of_ports
    vss_spec.mtu = self.mtu
    if self.nics:
        vss_spec.bridge = vim.host.VirtualSwitch.BondBridge(nicDevice=self.nics)
    if self.module.check_mode:
        results['msg'] = "vSwitch '%s' would be created" % self.switch
    else:
        try:
            self.network_mgr.AddVirtualSwitch(vswitchName=self.switch, spec=vss_spec)
            changed = False
            spec = self.find_vswitch_by_name(self.host_system, self.switch).spec
            if self.update_security_policy(spec, results):
                changed = True
            if self.update_teaming_policy(spec, results):
                changed = True
            if self.update_traffic_shaping_policy(spec, results):
                changed = True
            if changed:
                self.network_mgr.UpdateVirtualSwitch(vswitchName=self.switch, spec=spec)
            results['result'] = "vSwitch '%s' is created successfully" % self.switch
        except vim.fault.AlreadyExists as already_exists:
            results['result'] = 'vSwitch with name %s already exists: %s' % (self.switch, to_native(already_exists.msg))
        except vim.fault.ResourceInUse as resource_used:
            self.module.fail_json(msg="Failed to add vSwitch '%s' as physical network adapter being bridged is already in use: %s" % (self.switch, to_native(resource_used.msg)))
        except vim.fault.HostConfigFault as host_config_fault:
            self.module.fail_json(msg="Failed to add vSwitch '%s' due to host configuration fault : %s" % (self.switch, to_native(host_config_fault.msg)))
        except vmodl.fault.InvalidArgument as invalid_argument:
            self.module.fail_json(msg="Failed to add vSwitch '%s', this can be due to either of following : 1. vSwitch Name exceeds the maximum allowed length, 2. Number of ports specified falls out of valid range, 3. Network policy is invalid, 4. Beacon configuration is invalid : %s" % (self.switch, to_native(invalid_argument.msg)))
        except vmodl.fault.SystemError as system_error:
            self.module.fail_json(msg="Failed to add vSwitch '%s' due to : %s" % (self.switch, to_native(system_error.msg)))
        except Exception as generic_exc:
            self.module.fail_json(msg="Failed to add vSwitch '%s' due to generic exception : %s" % (self.switch, to_native(generic_exc)))
    results['changed'] = True
    self.module.exit_json(**results)