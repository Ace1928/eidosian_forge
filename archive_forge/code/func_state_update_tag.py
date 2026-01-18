from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def state_update_tag(self):
    """
        Update tag

        """
    changed = False
    tag_id = self.tag_obj.id
    results = dict(msg='Tag %s is unchanged.' % self.tag_name, tag_id=tag_id)
    tag_desc = self.tag_obj.description
    desired_tag_desc = self.params.get('tag_description')
    if tag_desc != desired_tag_desc:
        tag_update_spec = self.tag_service.UpdateSpec()
        tag_update_spec.description = desired_tag_desc
        try:
            self.tag_service.update(tag_id, tag_update_spec)
        except Error as error:
            self.module.fail_json(msg='%s' % self.get_error_message(error))
        results['msg'] = 'Tag %s updated.' % self.tag_name
        changed = True
    self.module.exit_json(changed=changed, tag_status=results)