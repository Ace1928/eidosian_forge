from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def configure_ha(self):
    """
        Manage HA Configuration

        """
    changed, result = (False, None)
    if self.check_ha_config_diff():
        if not self.module.check_mode:
            cluster_config_spec = vim.cluster.ConfigSpecEx()
            cluster_config_spec.dasConfig = vim.cluster.DasConfigInfo()
            cluster_config_spec.dasConfig.enabled = self.enable_ha
            if self.enable_ha:
                vm_tool_spec = vim.cluster.VmToolsMonitoringSettings()
                vm_tool_spec.enabled = True
                vm_tool_spec.vmMonitoring = self.params.get('ha_vm_monitoring')
                vm_tool_spec.failureInterval = self.params.get('ha_vm_failure_interval')
                vm_tool_spec.minUpTime = self.params.get('ha_vm_min_up_time')
                vm_tool_spec.maxFailures = self.params.get('ha_vm_max_failures')
                vm_tool_spec.maxFailureWindow = self.params.get('ha_vm_max_failure_window')
                das_vm_config = vim.cluster.DasVmSettings()
                das_vm_config.restartPriority = self.params.get('ha_restart_priority')
                das_vm_config.isolationResponse = self.host_isolation_response
                das_vm_config.vmToolsMonitoringSettings = vm_tool_spec
                das_vm_config.vmComponentProtectionSettings = vim.cluster.VmComponentProtectionSettings()
                das_vm_config.vmComponentProtectionSettings.vmStorageProtectionForAPD = self.params.get('apd_response')
                if self.params.get('apd_response') != 'disabled' and self.params.get('apd_response') != 'warning':
                    das_vm_config.vmComponentProtectionSettings.vmTerminateDelayForAPDSec = self.params.get('apd_delay')
                    das_vm_config.vmComponentProtectionSettings.vmReactionOnAPDCleared = self.params.get('apd_reaction')
                das_vm_config.vmComponentProtectionSettings.vmStorageProtectionForPDL = self.params.get('pdl_response')
                if self.params['apd_response'] != 'disabled' or self.params['pdl_response'] != 'disabled':
                    cluster_config_spec.dasConfig.vmComponentProtecting = 'enabled'
                else:
                    cluster_config_spec.dasConfig.vmComponentProtecting = 'disabled'
                cluster_config_spec.dasConfig.defaultVmSettings = das_vm_config
            cluster_config_spec.dasConfig.admissionControlEnabled = self.ha_admission_control
            if self.ha_admission_control:
                if self.params.get('slot_based_admission_control'):
                    cluster_config_spec.dasConfig.admissionControlPolicy = vim.cluster.FailoverLevelAdmissionControlPolicy()
                    policy = self.params.get('slot_based_admission_control')
                    cluster_config_spec.dasConfig.admissionControlPolicy.failoverLevel = policy.get('failover_level')
                elif self.params.get('reservation_based_admission_control'):
                    cluster_config_spec.dasConfig.admissionControlPolicy = vim.cluster.FailoverResourcesAdmissionControlPolicy()
                    policy = self.params.get('reservation_based_admission_control')
                    auto_compute_percentages = policy.get('auto_compute_percentages')
                    cluster_config_spec.dasConfig.admissionControlPolicy.autoComputePercentages = auto_compute_percentages
                    cluster_config_spec.dasConfig.admissionControlPolicy.failoverLevel = policy.get('failover_level')
                    if not auto_compute_percentages:
                        cluster_config_spec.dasConfig.admissionControlPolicy.cpuFailoverResourcesPercent = policy.get('cpu_failover_resources_percent')
                        cluster_config_spec.dasConfig.admissionControlPolicy.memoryFailoverResourcesPercent = policy.get('memory_failover_resources_percent')
                elif self.params.get('failover_host_admission_control'):
                    cluster_config_spec.dasConfig.admissionControlPolicy = vim.cluster.FailoverHostAdmissionControlPolicy()
                    policy = self.params.get('failover_host_admission_control')
                    cluster_config_spec.dasConfig.admissionControlPolicy.failoverHosts = self.get_failover_hosts()
            cluster_config_spec.dasConfig.hostMonitoring = self.params.get('ha_host_monitoring')
            cluster_config_spec.dasConfig.vmMonitoring = self.params.get('ha_vm_monitoring')
            if self.changed_advanced_settings:
                cluster_config_spec.dasConfig.option = self.changed_advanced_settings
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