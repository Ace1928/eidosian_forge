from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def build_vault_provider_spec(params):
    if params['state'] == 'absent':
        return {}
    client = arguments.get_spec_payload(params, 'address', 'token', 'version', 'max_retries')
    if params.get('tls'):
        client['tls'] = arguments.get_spec_payload(params['tls'], 'ca_cert', 'client_cert', 'client_key', 'cname')
    if params.get('timeout'):
        client['timeout'] = _format_seconds(params['timeout'])
    if params.get('rate_limit') or params.get('burst_limit'):
        client['rate_limiter'] = arguments.get_renamed_spec_payload(params, dict(rate_limit='limit', burst_limit='burst'))
    return dict(client=client)