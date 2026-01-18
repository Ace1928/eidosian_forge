from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_acl(self):
    """
        Return details about the cifs-share-access-control
        :param:
            name : Name of the cifs-share-access-control
        :return: Details about the cifs-share-access-control. None if not found.
        :rtype: dict
        """
    cifs_acl_iter = netapp_utils.zapi.NaElement('cifs-share-access-control-get-iter')
    cifs_acl_info = netapp_utils.zapi.NaElement('cifs-share-access-control')
    cifs_acl_info.add_new_child('share', self.parameters['share_name'])
    cifs_acl_info.add_new_child('user-or-group', self.parameters['user_or_group'])
    cifs_acl_info.add_new_child('vserver', self.parameters['vserver'])
    if self.parameters.get('type') is not None:
        cifs_acl_info.add_new_child('user-group-type', self.parameters['type'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(cifs_acl_info)
    cifs_acl_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(cifs_acl_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting cifs-share-access-control %s: %s' % (self.parameters['share_name'], to_native(error)))
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        cifs_acl = result.get_child_by_name('attributes-list').get_child_by_name('cifs-share-access-control')
        return_value = {'share': cifs_acl.get_child_content('share'), 'user-or-group': cifs_acl.get_child_content('user-or-group'), 'permission': cifs_acl.get_child_content('permission'), 'type': cifs_acl.get_child_content('user-group-type')}
    return return_value