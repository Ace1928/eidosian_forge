from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def create_or_update_project(self, module, project_name, namespace, options):
    changed = False
    project_options = {'name': project_name, 'description': options['description'], 'issues_enabled': options['issues_enabled'], 'merge_requests_enabled': options['merge_requests_enabled'], 'merge_method': options['merge_method'], 'wiki_enabled': options['wiki_enabled'], 'snippets_enabled': options['snippets_enabled'], 'visibility': options['visibility'], 'lfs_enabled': options['lfs_enabled'], 'allow_merge_on_skipped_pipeline': options['allow_merge_on_skipped_pipeline'], 'only_allow_merge_if_all_discussions_are_resolved': options['only_allow_merge_if_all_discussions_are_resolved'], 'only_allow_merge_if_pipeline_succeeds': options['only_allow_merge_if_pipeline_succeeds'], 'packages_enabled': options['packages_enabled'], 'remove_source_branch_after_merge': options['remove_source_branch_after_merge'], 'squash_option': options['squash_option'], 'ci_config_path': options['ci_config_path'], 'shared_runners_enabled': options['shared_runners_enabled'], 'builds_access_level': options['builds_access_level'], 'forking_access_level': options['forking_access_level'], 'container_registry_access_level': options['container_registry_access_level'], 'releases_access_level': options['releases_access_level'], 'environments_access_level': options['environments_access_level'], 'feature_flags_access_level': options['feature_flags_access_level'], 'infrastructure_access_level': options['infrastructure_access_level'], 'monitor_access_level': options['monitor_access_level'], 'security_and_compliance_access_level': options['security_and_compliance_access_level']}
    if LooseVersion(self._gitlab.version()[0]) < LooseVersion('14'):
        project_options['tag_list'] = options['topics']
    else:
        project_options['topics'] = options['topics']
    if self.project_object is None:
        if options['default_branch'] and (not options['initialize_with_readme']):
            module.fail_json(msg='Param default_branch need param initialize_with_readme set to true')
        project_options.update({'path': options['path'], 'import_url': options['import_url']})
        if options['initialize_with_readme']:
            project_options['initialize_with_readme'] = options['initialize_with_readme']
            if options['default_branch']:
                project_options['default_branch'] = options['default_branch']
        project_options = self.get_options_with_value(project_options)
        project = self.create_project(namespace, project_options)
        if options['avatar_path']:
            try:
                project.avatar = open(options['avatar_path'], 'rb')
            except IOError as e:
                self._module.fail_json(msg='Cannot open {0}: {1}'.format(options['avatar_path'], e))
        changed = True
    else:
        if options['default_branch']:
            project_options['default_branch'] = options['default_branch']
        changed, project = self.update_project(self.project_object, project_options)
    self.project_object = project
    if changed:
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Successfully created or updated the project %s' % project_name)
        try:
            project.save()
        except Exception as e:
            self._module.fail_json(msg='Failed update project: %s ' % e)
        return True
    return False