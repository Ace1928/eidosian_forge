from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def delete_dr_groups(self, dr_ids):
    for dr_id in dr_ids:
        api = 'cluster/metrocluster/dr-groups/' + str(dr_id)
        message, error = self.rest_api.delete(api)
        if error:
            self.module.fail_json(msg=error)
        message, error = self.rest_api.wait_on_job(message['job'])
        if error:
            self.module.fail_json(msg='%s' % error)