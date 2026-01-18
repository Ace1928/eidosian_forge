from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_fdspt(self):
    """
        Get File Directory Security Policy Task
        """
    api = 'private/cli/vserver/security/file-directory/policy/task'
    query = {'policy-name': self.parameters['name'], 'path': self.parameters['path'], 'fields': 'vserver,ntfs-mode,ntfs-sd,security-type,access-control,index-num'}
    message, error = self.rest_api.get(api, query)
    records, error = rrh.check_for_0_or_1_records(api, message, error)
    if error:
        self.module.fail_json(msg=error)
    if records:
        if 'ntfs_sd' not in records:
            records['ntfs_sd'] = []
    return records if records else None