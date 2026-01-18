from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, _load_params
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def create_pipeline_variable(module, bitbucket):
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['pipeline-variable-list'].format(workspace=module.params['workspace'], repo_slug=module.params['repository']), method='POST', data={'key': module.params['name'], 'value': module.params['value'], 'secured': module.params['secured']})
    if info['status'] != 201:
        module.fail_json(msg='Failed to create pipeline variable `{name}`: {info}'.format(name=module.params['name'], info=info))