from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible.module_utils.six import iteritems
from ansible_collections.community.network.plugins.module_utils.network.ftd.configuration import BaseConfigurationResource, ParamName
from ansible_collections.community.network.plugins.module_utils.network.ftd.device import assert_kick_is_installed, FtdPlatformFactory, FtdModel
from ansible_collections.community.network.plugins.module_utils.network.ftd.operation import FtdOperations, get_system_info
def check_that_model_is_supported(module, platform_model):
    if platform_model not in FtdModel.supported_models():
        module.fail_json(msg="Platform model '%s' is not supported by this module." % platform_model)