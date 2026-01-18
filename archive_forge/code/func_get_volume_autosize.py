from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_volume_autosize(self):
    """
        Get volume_autosize information from the ONTAP system
        :return:
        """
    if self.use_rest:
        query = {'name': self.parameters['volume'], 'svm.name': self.parameters['vserver'], 'fields': 'autosize,uuid'}
        api = 'storage/volumes'
        response, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error is not None:
            self.module.fail_json(msg='Error fetching volume autosize info for %s: %s' % (self.parameters['volume'], error))
        if response:
            return self._create_get_volume_return(response['autosize'], response['uuid'])
        self.module.fail_json(msg='Error fetching volume autosize info for %s: volume not found for vserver %s.' % (self.parameters['volume'], self.parameters['vserver']))
    else:
        volume_autosize_info = netapp_utils.zapi.NaElement('volume-autosize-get')
        volume_autosize_info.add_new_child('volume', self.parameters['volume'])
        try:
            result = self.server.invoke_successfully(volume_autosize_info, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching volume autosize info for %s: %s.' % (self.parameters['volume'], to_native(error)), exception=traceback.format_exc())
        return self._create_get_volume_return(result)