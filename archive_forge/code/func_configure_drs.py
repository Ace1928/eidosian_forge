from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def configure_drs(self):
    """
        Manage DRS configuration
        """
    changed, result = (False, None)
    if self.check_drs_config_diff():
        if not self.module.check_mode:
            cluster_config_spec = vim.cluster.ConfigSpecEx()
            cluster_config_spec.drsConfig = vim.cluster.DrsConfigInfo()
            cluster_config_spec.proactiveDrsConfig = vim.cluster.ProactiveDrsConfigInfo()
            cluster_config_spec.drsConfig.enabled = self.enable_drs
            cluster_config_spec.drsConfig.enableVmBehaviorOverrides = self.params.get('drs_enable_vm_behavior_overrides')
            cluster_config_spec.drsConfig.defaultVmBehavior = self.params.get('drs_default_vm_behavior')
            cluster_config_spec.drsConfig.vmotionRate = self.drs_vmotion_rate
            cluster_config_spec.proactiveDrsConfig.enabled = self.params.get('predictive_drs')
            if self.changed_advanced_settings:
                cluster_config_spec.drsConfig.option = self.changed_advanced_settings
            try:
                task = self.cluster.ReconfigureComputeResource_Task(cluster_config_spec, True)
                changed, result = wait_for_task(task)
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