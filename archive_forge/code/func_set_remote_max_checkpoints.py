from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible_collections.community.network.plugins.module_utils.network.sros.sros import load_config, get_config, sros_argument_spec, check_args
def set_remote_max_checkpoints(module, commands):
    value = module.params['remote_max_checkpoints']
    if not 1 <= value <= 50:
        module.fail_json(msg='remote_max_checkpoints must be between 1 and 50')
    commands.append('configure system rollback remote-max-checkpoints %s' % value)