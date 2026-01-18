from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def create_or_update_runner(self, description, options):
    changed = False
    arguments = {'locked': options['locked'], 'run_untagged': options['run_untagged'], 'maximum_timeout': options['maximum_timeout'], 'tag_list': options['tag_list']}
    if options.get('paused') is not None:
        arguments['paused'] = options['paused']
    else:
        arguments['active'] = options['active']
    if options.get('access_level') is not None:
        arguments['access_level'] = options['access_level']
    if self.runner_object is None:
        arguments['description'] = description
        if options.get('registration_token') is not None:
            arguments['token'] = options['registration_token']
        elif options.get('group') is not None:
            arguments['runner_type'] = 'group_type'
            arguments['group_id'] = options['group']
        elif options.get('project') is not None:
            arguments['runner_type'] = 'project_type'
            arguments['project_id'] = options['project']
        else:
            arguments['runner_type'] = 'instance_type'
        access_level_on_creation = self._module.params['access_level_on_creation']
        if not access_level_on_creation:
            arguments.pop('access_level', None)
        runner = self.create_runner(arguments)
        changed = True
    else:
        changed, runner = self.update_runner(self.runner_object, arguments)
        if changed:
            if self._module.check_mode:
                self._module.exit_json(changed=True, msg='Successfully updated the runner %s' % description)
            try:
                runner.save()
            except Exception as e:
                self._module.fail_json(msg='Failed to update runner: %s ' % to_native(e))
    self.runner_object = runner
    return changed