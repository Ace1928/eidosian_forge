from __future__ import absolute_import, division, print_function
import re
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import KatelloEntityAnsibleModule, _flatten_entity
def promote_content_view_version(module, content_view_version, environments, force, force_yum_metadata_regeneration):
    current_environment_ids = {environment['id'] for environment in content_view_version['environments']}
    desired_environment_ids = {environment['id'] for environment in environments}
    promote_to_environment_ids = list(desired_environment_ids - current_environment_ids)
    if promote_to_environment_ids:
        payload = {'id': content_view_version['id'], 'environment_ids': promote_to_environment_ids, 'force': force, 'force_yum_metadata_regeneration': force_yum_metadata_regeneration}
        module.resource_action('content_view_versions', 'promote', params=payload)