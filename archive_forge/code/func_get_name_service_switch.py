from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_name_service_switch(self):
    """
        get current name service switch config
        :return: dict of current name service switch
        """
    if self.use_rest:
        return self.get_name_service_switch_rest()
    nss_iter = netapp_utils.zapi.NaElement('nameservice-nsswitch-get-iter')
    nss_info = netapp_utils.zapi.NaElement('namservice-nsswitch-config-info')
    db_type = netapp_utils.zapi.NaElement('nameservice-database')
    db_type.set_content(self.parameters['database_type'])
    query = netapp_utils.zapi.NaElement('query')
    nss_info.add_child_elem(db_type)
    query.add_child_elem(nss_info)
    nss_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(nss_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching name service switch info for %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        nss_sources = result.get_child_by_name('attributes-list').get_child_by_name('namservice-nsswitch-config-info').get_child_by_name('nameservice-sources')
        if nss_sources:
            sources = [sources.get_content() for sources in nss_sources.get_children()]
            return_value = {'sources': sources}
        else:
            return_value = {'sources': []}
    return return_value