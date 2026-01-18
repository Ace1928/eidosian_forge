from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Cli
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def get_resource_connection(module):
    if hasattr(module, '_connection'):
        return module._connection
    capabilities = get_capabilities(module)
    network_api = capabilities.get('network_api')
    if network_api == 'netconf':
        module._connection = NetconfConnection(module._socket_path)
    elif network_api == 'local':
        module._connection = LocalResourceConnection(module)
    else:
        module._connection = Connection(module._socket_path)
    return module._connection