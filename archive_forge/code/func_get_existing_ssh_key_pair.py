from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def get_existing_ssh_key_pair(module, bitbucket):
    """
    Retrieves an existing ssh key pair from repository
    specified in module param `repository`

    :param module: instance of the :class:`AnsibleModule`
    :param bitbucket: instance of the :class:`BitbucketHelper`
    :return: existing key pair or None if not found
    :rtype: dict or None

    Return example::

        {
            "public_key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ...2E8HAeT",
            "type": "pipeline_ssh_key_pair"
        }
    """
    api_url = BITBUCKET_API_ENDPOINTS['ssh-key-pair'].format(workspace=module.params['workspace'], repo_slug=module.params['repository'])
    info, content = bitbucket.request(api_url=api_url, method='GET')
    if info['status'] == 404:
        return None
    return content