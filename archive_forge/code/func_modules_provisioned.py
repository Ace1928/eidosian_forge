from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
def modules_provisioned(client):
    """Returns a list of all provisioned modules

    Args:
        client: Client connection to the BIG-IP

    Returns:
        A list of provisioned modules in their short name for.
        For example, ['afm', 'asm', 'ltm']
    """
    uri = 'https://{0}:{1}/mgmt/tm/sys/provision'.format(client.provider['server'], client.provider['server_port'])
    resp = client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if 'items' not in response:
        return []
    return [x['name'] for x in response['items'] if x['level'] != 'none']