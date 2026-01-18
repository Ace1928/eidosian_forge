from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def create_deploy_key(module, bitbucket):
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['deploy-key-list'].format(workspace=module.params['workspace'], repo_slug=module.params['repository']), method='POST', data={'key': module.params['key'], 'label': module.params['label']})
    if info['status'] == 404:
        module.fail_json(msg=error_messages['invalid_workspace_or_repo'])
    if info['status'] == 403:
        module.fail_json(msg=error_messages['required_permission'])
    if info['status'] == 400:
        module.fail_json(msg=error_messages['invalid_key'])
    if info['status'] != 200:
        module.fail_json(msg='Failed to create deploy key `{label}`: {info}'.format(label=module.params['label'], info=info))