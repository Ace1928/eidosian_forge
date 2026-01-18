from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible.module_utils.six import iteritems
from ansible_collections.community.network.plugins.module_utils.network.ftd.configuration import BaseConfigurationResource, ParamName
from ansible_collections.community.network.plugins.module_utils.network.ftd.device import assert_kick_is_installed, FtdPlatformFactory, FtdModel
from ansible_collections.community.network.plugins.module_utils.network.ftd.operation import FtdOperations, get_system_info
def check_that_update_is_needed(module, system_info):
    target_ftd_version = module.params['image_version']
    if not module.params['force_install'] and target_ftd_version == system_info['softwareVersion']:
        module.exit_json(changed=False, msg='FTD already has %s version of software installed.' % target_ftd_version)