from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_volume_autosize(self, uuid):
    """
        Modify a Volumes autosize
        :return:
        """
    if self.use_rest:
        autosize = {}
        if self.parameters.get('mode'):
            autosize['mode'] = self.parameters['mode']
        if self.parameters.get('grow_threshold_percent'):
            autosize['grow_threshold'] = self.parameters['grow_threshold_percent']
        if self.parameters.get('maximum_size'):
            autosize['maximum'] = self.parameters['maximum_size']
        if self.parameters.get('minimum_size'):
            autosize['minimum'] = self.parameters['minimum_size']
        if self.parameters.get('shrink_threshold_percent'):
            autosize['shrink_threshold'] = self.parameters['shrink_threshold_percent']
        if not autosize:
            return
        api = 'storage/volumes'
        body = {'autosize': autosize}
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
        if error is not None:
            self.module.fail_json(msg='Error modifying volume autosize for %s: %s' % (self.parameters['volume'], error))
    else:
        volume_autosize_info = netapp_utils.zapi.NaElement('volume-autosize-set')
        volume_autosize_info.add_new_child('volume', self.parameters['volume'])
        if self.parameters.get('mode'):
            volume_autosize_info.add_new_child('mode', self.parameters['mode'])
        if self.parameters.get('grow_threshold_percent'):
            volume_autosize_info.add_new_child('grow-threshold-percent', str(self.parameters['grow_threshold_percent']))
        if self.parameters.get('increment_size'):
            volume_autosize_info.add_new_child('increment-size', self.parameters['increment_size'])
        if self.parameters.get('reset') is not None:
            volume_autosize_info.add_new_child('reset', str(self.parameters['reset']))
        if self.parameters.get('maximum_size'):
            volume_autosize_info.add_new_child('maximum-size', self.parameters['maximum_size'])
        if self.parameters.get('minimum_size'):
            volume_autosize_info.add_new_child('minimum-size', self.parameters['minimum_size'])
        if self.parameters.get('shrink_threshold_percent'):
            volume_autosize_info.add_new_child('shrink-threshold-percent', str(self.parameters['shrink_threshold_percent']))
        try:
            self.server.invoke_successfully(volume_autosize_info, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying volume autosize for %s: %s.' % (self.parameters['volume'], to_native(error)), exception=traceback.format_exc())