from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_local_group_member(self):
    """
        Retrieves local users, Active Directory users and
        Active Directory groups which are members of the specified local group and SVM.
        """
    return_value = None
    if self.use_rest:
        self.get_cifs_local_group_rest()
        api = 'protocols/cifs/local-groups/%s/%s/members' % (self.svm_uuid, self.sid)
        query = {'name': self.parameters['member'], 'svm.name': self.parameters['vserver'], 'fields': 'name'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error getting CIFS local group members for group %s on vserver %s: %s' % (self.parameters['group'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if record:
            return {'member': self.na_helper.safe_get(record, ['name'])}
        return record
    else:
        group_members_get_iter = netapp_utils.zapi.NaElement('cifs-local-group-members-get-iter')
        group_members_info = netapp_utils.zapi.NaElement('cifs-local-group-members')
        group_members_info.add_new_child('group-name', self.parameters['group'])
        group_members_info.add_new_child('vserver', self.parameters['vserver'])
        group_members_info.add_new_child('member', self.parameters['member'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(group_members_info)
        group_members_get_iter.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(group_members_get_iter, True)
            if result.get_child_by_name('attributes-list'):
                group_member_policy_attributes = result['attributes-list']['cifs-local-group-members']
                return_value = {'group': group_member_policy_attributes['group-name'], 'member': group_member_policy_attributes['member'], 'vserver': group_member_policy_attributes['vserver']}
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting CIFS local group members for group %s on vserver %s: %s' % (self.parameters['group'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        return return_value