from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible.module_utils.six import iteritems
from ansible_collections.community.network.plugins.module_utils.network.ftd.configuration import BaseConfigurationResource, ParamName
from ansible_collections.community.network.plugins.module_utils.network.ftd.device import assert_kick_is_installed, FtdPlatformFactory, FtdModel
from ansible_collections.community.network.plugins.module_utils.network.ftd.operation import FtdOperations, get_system_info
def check_required_params_for_local_connection(module, params):
    missing_params = [k for k, v in iteritems(params) if k in REQUIRED_PARAMS_FOR_LOCAL_CONNECTION and v is None]
    if missing_params:
        message = "The following parameters are mandatory when the module is used with 'local' connection: %s." % ', '.join(sorted(missing_params))
        module.fail_json(msg=message)