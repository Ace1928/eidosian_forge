from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_get(self):
    """
        Get current FlexCache relations
        :return: Dictionary of current FlexCache details if query successful, else None
        """
    if self.use_rest:
        api = 'storage/flexcache/flexcaches'
        query = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
        if 'origin_cluster' in self.parameters:
            query['origin.cluster.name'] = self.parameters['origin_cluster']
        fields = 'svm,name,uuid,path'
        flexcache, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
        self.na_helper.fail_on_error(error)
        if flexcache is None:
            return None
        return dict(vserver=flexcache['svm']['name'], name=flexcache['name'], uuid=flexcache['uuid'], junction_path=flexcache.get('path'))
    flexcache_get_iter = self.flexcache_get_iter()
    flex_info = {}
    try:
        result = self.server.invoke_successfully(flexcache_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching FlexCache info: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        flexcache_info = result.get_child_by_name('attributes-list').get_child_by_name('flexcache-info')
        flex_info['origin_cluster'] = flexcache_info.get_child_content('origin-cluster')
        flex_info['origin_volume'] = flexcache_info.get_child_content('origin-volume')
        flex_info['origin_vserver'] = flexcache_info.get_child_content('origin-vserver')
        flex_info['size'] = flexcache_info.get_child_content('size')
        flex_info['name'] = flexcache_info.get_child_content('volume')
        flex_info['vserver'] = flexcache_info.get_child_content('vserver')
        return flex_info
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 1:
        msg = 'Multiple records found for %s:' % self.parameters['name']
        self.module.fail_json(msg='Error fetching FlexCache info: %s' % msg)
    return None