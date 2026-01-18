from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def create_or_update_deploy_key(self, project, key_title, key_key, options):
    changed = False
    if self.deploy_key_object and self.deploy_key_object.key != key_key:
        if not self._module.check_mode:
            self.deploy_key_object.delete()
        self.deploy_key_object = None
    if self.deploy_key_object is None:
        deploy_key = self.create_deploy_key(project, {'title': key_title, 'key': key_key, 'can_push': options['can_push']})
        changed = True
    else:
        changed, deploy_key = self.update_deploy_key(self.deploy_key_object, {'title': key_title, 'can_push': options['can_push']})
    self.deploy_key_object = deploy_key
    if changed:
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Successfully created or updated the deploy key %s' % key_title)
        try:
            deploy_key.save()
        except Exception as e:
            self._module.fail_json(msg='Failed to update deploy key: %s ' % e)
        return True
    else:
        return False