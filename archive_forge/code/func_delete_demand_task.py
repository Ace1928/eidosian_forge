from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_demand_task(self):
    """
        Delete a Demand Task"
        :return:
        """
    if self.use_rest:
        return self.delete_demand_task_rest()
    demand_task_obj = netapp_utils.zapi.NaElement('vscan-on-demand-task-delete')
    demand_task_obj.add_new_child('task-name', self.parameters['task_name'])
    try:
        self.server.invoke_successfully(demand_task_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting on demand task, %s: %s' % (self.parameters['task_name'], to_native(error)), exception=traceback.format_exc())