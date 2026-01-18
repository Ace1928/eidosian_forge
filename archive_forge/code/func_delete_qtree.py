from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_qtree(self):
    """
        Delete a qtree
        """
    if self.use_rest:
        api = 'storage/qtrees/%s' % self.volume_uuid
        query = {'return_timeout': 3}
        response, error = rest_generic.delete_async(self.rest_api, api, self.qid, query)
        if self.parameters['wait_for_completion']:
            dummy, error = rrh.check_for_error_and_job_results(api, response, error, self.rest_api)
        if error:
            self.module.fail_json(msg='Error deleting qtree %s: %s' % (self.parameters['name'], error))
    else:
        path = '/vol/%s/%s' % (self.parameters['flexvol_name'], self.parameters['name'])
        options = {'qtree': path}
        if self.parameters['force_delete']:
            options['force'] = 'true'
        qtree_delete = netapp_utils.zapi.NaElement.create_node_with_children('qtree-delete', **options)
        try:
            self.server.invoke_successfully(qtree_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting qtree %s: %s' % (path, to_native(error)), exception=traceback.format_exc())