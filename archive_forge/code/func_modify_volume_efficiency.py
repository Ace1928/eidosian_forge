from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_volume_efficiency(self, body=None):
    """
        Modifies volume efficiency settings for a given volume by path
        """
    if self.use_rest:
        if not body:
            return
        dummy, error = rest_generic.patch_async(self.rest_api, 'storage/volumes', self.volume_uuid, body)
        if error:
            if 'Unexpected argument "storage_efficiency_mode".' in error:
                error = 'cannot modify storage_efficiency mode in non AFF platform.'
            if 'not authorized' in error:
                error = '%s user is not authorized to modify volume efficiency' % self.parameters.get('username')
            self.module.fail_json(msg='Error in volume/efficiency patch: %s' % error)
    else:
        sis_config_obj = netapp_utils.zapi.NaElement('sis-set-config')
        sis_config_obj.add_new_child('path', self.parameters['path'])
        if 'schedule' in self.parameters:
            sis_config_obj.add_new_child('schedule', self.parameters['schedule'])
        if 'policy' in self.parameters:
            sis_config_obj.add_new_child('policy-name', self.parameters['policy'])
        if 'enable_compression' in self.parameters:
            sis_config_obj.add_new_child('enable-compression', self.na_helper.get_value_for_bool(False, self.parameters['enable_compression']))
        if 'enable_inline_compression' in self.parameters:
            sis_config_obj.add_new_child('enable-inline-compression', self.na_helper.get_value_for_bool(False, self.parameters['enable_inline_compression']))
        if 'enable_inline_dedupe' in self.parameters:
            sis_config_obj.add_new_child('enable-inline-dedupe', self.na_helper.get_value_for_bool(False, self.parameters['enable_inline_dedupe']))
        if 'enable_data_compaction' in self.parameters:
            sis_config_obj.add_new_child('enable-data-compaction', self.na_helper.get_value_for_bool(False, self.parameters['enable_data_compaction']))
        if 'enable_cross_volume_inline_dedupe' in self.parameters:
            sis_config_obj.add_new_child('enable-cross-volume-inline-dedupe', self.na_helper.get_value_for_bool(False, self.parameters['enable_cross_volume_inline_dedupe']))
        if 'enable_cross_volume_background_dedupe' in self.parameters:
            sis_config_obj.add_new_child('enable-cross-volume-background-dedupe', self.na_helper.get_value_for_bool(False, self.parameters['enable_cross_volume_background_dedupe']))
        try:
            self.server.invoke_successfully(sis_config_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying storage efficiency for path %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())