from __future__ import absolute_import, division, print_function
from collections import defaultdict
from ansible.module_utils.basic import to_text, AnsibleModule
def get_consul_client(configuration):
    """
    Gets a Consul client for the given configuration.

    Does not check if the Consul client can connect.
    :param configuration: the run configuration
    :return: Consul client
    """
    token = configuration.management_token
    if token is None:
        token = configuration.token
    if token is None:
        raise AssertionError('Expecting the management token to always be set')
    return consul.Consul(host=configuration.host, port=configuration.port, scheme=configuration.scheme, verify=configuration.validate_certs, token=token)