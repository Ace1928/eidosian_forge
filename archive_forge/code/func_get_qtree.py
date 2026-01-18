from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_qtree(self, name=None):
    """
        Checks if the qtree exists.
        :param:
            name : qtree name
        :return:
            Details about the qtree
            False if qtree is not found
        :rtype: bool
        """
    if name is None:
        name = self.parameters['name']
    if self.use_rest:
        api = 'storage/qtrees'
        query = {'fields': 'export_policy,unix_permissions,security_style,volume', 'svm.name': self.parameters['vserver'], 'volume': self.parameters['flexvol_name'], 'name': '"' + name + '"'}
        if 'unix_user' in self.parameters:
            query['fields'] += ',user.name'
        if 'unix_group' in self.parameters:
            query['fields'] += ',group.name'
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            msg = 'Error fetching qtree: %s' % error
            self.module.fail_json(msg=msg)
        if record:
            self.volume_uuid = record['volume']['uuid']
            self.qid = str(record['id'])
            return {'name': record['name'], 'export_policy': self.na_helper.safe_get(record, ['export_policy', 'name']), 'security_style': self.na_helper.safe_get(record, ['security_style']), 'unix_permissions': str(self.na_helper.safe_get(record, ['unix_permissions'])), 'unix_user': self.na_helper.safe_get(record, ['user', 'name']), 'unix_group': self.na_helper.safe_get(record, ['group', 'name'])}
        return None
    qtree_list_iter = netapp_utils.zapi.NaElement('qtree-list-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('qtree-info', **{'vserver': self.parameters['vserver'], 'volume': self.parameters['flexvol_name'], 'qtree': name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    qtree_list_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(qtree_list_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching qtree: %s' % to_native(error), exception=traceback.format_exc())
    return_q = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        return_q = {'export_policy': result['attributes-list']['qtree-info']['export-policy'], 'oplocks': result['attributes-list']['qtree-info']['oplocks'], 'security_style': result['attributes-list']['qtree-info']['security-style']}
        value = self.na_helper.safe_get(result, ['attributes-list', 'qtree-info', 'mode'])
        return_q['unix_permissions'] = value if value is not None else ''
    return return_q