from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def add_task_to_policy(self):
    policy_add_task = netapp_utils.zapi.NaElement('file-directory-security-policy-task-add')
    policy_add_task.add_new_child('path', self.parameters['path'])
    policy_add_task.add_new_child('policy-name', self.parameters['policy_name'])
    if self.parameters.get('access_control') is not None:
        policy_add_task.add_new_child('access-control', self.parameters['access_control'])
    if self.parameters.get('ntfs_mode') is not None:
        policy_add_task.add_new_child('ntfs-mode', self.parameters['ntfs_mode'])
    if self.parameters.get('ntfs_sd') is not None:
        ntfs_sds = netapp_utils.zapi.NaElement('ntfs-sd')
        for ntfs_sd in self.parameters['ntfs_sd']:
            ntfs_sds.add_new_child('file-security-ntfs-sd', ntfs_sd)
        policy_add_task.add_child_elem(ntfs_sds)
    if self.parameters.get('security_type') is not None:
        policy_add_task.add_new_child('security-type', self.parameters['security_type'])
    try:
        self.server.invoke_successfully(policy_add_task, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error adding task to file-directory policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())