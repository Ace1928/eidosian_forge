from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def configure_dpm(self):
    """
        Manage DRS configuration

        """
    changed, result = self.check_dpm_config_diff()
    if changed:
        if not self.module.check_mode:
            cluster_config_spec = vim.cluster.ConfigSpecEx()
            cluster_config_spec.dpmConfig = vim.cluster.DpmConfigInfo()
            cluster_config_spec.dpmConfig.enabled = self.enable_dpm
            cluster_config_spec.dpmConfig.defaultDpmBehavior = self.default_dpm_behaviour
            cluster_config_spec.dpmConfig.hostPowerActionRate = self.host_power_action_rate
            try:
                task = self.cluster.ReconfigureComputeResource_Task(cluster_config_spec, True)
                changed = wait_for_task(task)[0]
            except vmodl.RuntimeFault as runtime_fault:
                self.module.fail_json(msg=to_native(runtime_fault.msg))
            except vmodl.MethodFault as method_fault:
                self.module.fail_json(msg=to_native(method_fault.msg))
            except TaskError as task_e:
                self.module.fail_json(msg=to_native(task_e))
            except Exception as generic_exc:
                self.module.fail_json(msg='Failed to update cluster due to generic exception %s' % to_native(generic_exc))
        else:
            changed = True
    self.module.exit_json(changed=changed, result=result)