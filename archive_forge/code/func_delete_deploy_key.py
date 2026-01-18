from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def delete_deploy_key(module, bitbucket, key_id):
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['deploy-key-detail'].format(workspace=module.params['workspace'], repo_slug=module.params['repository'], key_id=key_id), method='DELETE')
    if info['status'] == 404:
        module.fail_json(msg=error_messages['invalid_workspace_or_repo'])
    if info['status'] == 403:
        module.fail_json(msg=error_messages['required_permission'])
    if info['status'] != 204:
        module.fail_json(msg='Failed to delete deploy key `{label}`: {info}'.format(label=module.params['label'], info=info))