from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def server_configurable_configlet(module, sw_info):
    """ Check CVP that the user specified switch has a configlet assigned to
        it that Ansible is allowed to edit.

    :param module: Ansible module with parameters and client connection.
    :param sw_info: Dict of switch info.
    :return: Dict of configlet information or None.
    """
    configurable_configlet = None
    configlet_name = module.params['switch_name'] + '-server'
    switch_configlets = module.client.api.get_configlets_by_device_id(sw_info['key'])
    for configlet in switch_configlets:
        if configlet['name'] == configlet_name:
            configurable_configlet = configlet
    return configurable_configlet