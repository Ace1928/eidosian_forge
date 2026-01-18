from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def get_existing_known_host(module, bitbucket):
    """
    Search for a host in Bitbucket pipelines known hosts
    with the name specified in module param `name`

    :param module: instance of the :class:`AnsibleModule`
    :param bitbucket: instance of the :class:`BitbucketHelper`
    :return: existing host or None if not found
    :rtype: dict or None

    Return example::

        {
            'type': 'pipeline_known_host',
            'uuid': '{21cc0590-bebe-4fae-8baf-03722704119a7}'
            'hostname': 'bitbucket.org',
            'public_key': {
                'type': 'pipeline_ssh_public_key',
                'md5_fingerprint': 'md5:97:8c:1b:f2:6f:14:6b:4b:3b:ec:aa:46:46:74:7c:40',
                'sha256_fingerprint': 'SHA256:zzXQOXSFBEiUtuE8AikoYKwbHaxvSc0ojez9YXaGp1A',
                'key_type': 'ssh-rsa',
                'key': 'AAAAB3NzaC1yc2EAAAABIwAAAQEAubiN81eDcafrgMeLzaFPsw2kN...seeFVBoGqzHM9yXw=='
            },
        }
    """
    content = {'next': BITBUCKET_API_ENDPOINTS['known-host-list'].format(workspace=module.params['workspace'], repo_slug=module.params['repository'])}
    while 'next' in content:
        info, content = bitbucket.request(api_url=content['next'], method='GET')
        if info['status'] == 404:
            module.fail_json(msg='Invalid `repository` or `workspace`.')
        if info['status'] != 200:
            module.fail_json(msg='Failed to retrieve list of known hosts: {0}'.format(info))
        host = next(filter(lambda v: v['hostname'] == module.params['name'], content['values']), None)
        if host is not None:
            return host
    return None