from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def configure_vCLS(self):
    """
        Manage DRS configuration

        """
    result = None
    changed, toAddAllowedDatastores, toRemoveAllowedDatastores = self.check_vCLS_config_diff()
    if changed:
        if not self.module.check_mode:
            cluster_config_spec = vim.cluster.ConfigSpecEx()
            cluster_config_spec.systemVMsConfig = vim.cluster.SystemVMsConfigSpec()
            cluster_config_spec.systemVMsConfig.allowedDatastores = []
            for ds_name in toAddAllowedDatastores:
                specSystemVMsConfigAllowedDatastore = vim.cluster.DatastoreUpdateSpec()
                specSystemVMsConfigAllowedDatastore.datastore = find_datastore_by_name(self.content, ds_name, self.datacenter)
                specSystemVMsConfigAllowedDatastore.operation = 'add'
                cluster_config_spec.systemVMsConfig.allowedDatastores.append(specSystemVMsConfigAllowedDatastore)
            for ds_name in toRemoveAllowedDatastores:
                specSystemVMsConfigAllowedDatastore = vim.cluster.DatastoreUpdateSpec()
                specSystemVMsConfigAllowedDatastore.removeKey = find_datastore_by_name(self.content, ds_name, self.datacenter)
                specSystemVMsConfigAllowedDatastore.operation = 'remove'
                cluster_config_spec.systemVMsConfig.allowedDatastores.append(specSystemVMsConfigAllowedDatastore)
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
    results = dict(changed=changed)
    results['result'] = result
    results['Added_AllowedDatastores'] = toAddAllowedDatastores
    results['Removed_AllowedDatastores'] = toRemoveAllowedDatastores
    self.module.exit_json(**results)