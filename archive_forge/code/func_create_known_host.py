from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def create_known_host(module, bitbucket):
    hostname = module.params['name']
    key_param = module.params['key']
    if key_param is None:
        key_type, key = get_host_key(module, hostname)
    elif ' ' in key_param:
        key_type, key = key_param.split(' ', 1)
    else:
        module.fail_json(msg=error_messages['unknown_key_type'])
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['known-host-list'].format(workspace=module.params['workspace'], repo_slug=module.params['repository']), method='POST', data={'hostname': hostname, 'public_key': {'key_type': key_type, 'key': key}})
    if info['status'] == 404:
        module.fail_json(msg=error_messages['invalid_params'])
    if info['status'] != 201:
        module.fail_json(msg='Failed to create known host `{hostname}`: {info}'.format(hostname=module.params['hostname'], info=info))