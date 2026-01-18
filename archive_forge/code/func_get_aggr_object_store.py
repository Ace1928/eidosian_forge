from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_aggr_object_store(self):
    """
        Fetch details if object store config exists.
        :return:
            Dictionary of current details if object store config found
            None if object store config is not found
        """
    if self.use_rest:
        api = 'cloud/targets'
        query = {'name': self.parameters['name']}
        fields = ','.join(self.rest_get_fields)
        fields += ',uuid'
        record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
        if error:
            self.module.fail_json(msg='Error %s' % error)
        return record
    else:
        aggr_object_store_get_iter = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-config-get', **{'object-store-name': self.parameters['name']})
        try:
            result = self.server.invoke_successfully(aggr_object_store_get_iter, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            if to_native(error.code) == '15661':
                return None
            else:
                self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        info = self.na_helper.safe_get(result, ['attributes', 'aggr-object-store-config-info'])
        if info:
            zapi_to_rest = {'access_key': dict(key_list=['access-key'], convert_to=str), 'certificate_validation_enabled': dict(key_list=['is-certificate-validation-enabled'], convert_to=bool), 'container': dict(key_list=['s3-name'], convert_to=str), 'name': dict(key_list=['object-store-name'], convert_to=str), 'port': dict(key_list=['port'], convert_to=int), 'provider_type': dict(key_list=['provider-type'], convert_to=str), 'ssl_enabled': dict(key_list=['ssl-enabled'], convert_to=bool), 'server': dict(key_list=['server'], convert_to=str)}
            results = {}
            self.na_helper.zapi_get_attrs(info, zapi_to_rest, results)
            return results
        return None