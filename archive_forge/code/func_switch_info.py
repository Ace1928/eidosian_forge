from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def switch_info(module):
    """ Get dictionary of switch info from CVP.

    :param module: Ansible module with parameters and client connection.
    :return: Dict of switch info from CVP or exit with failure if no
             info for device is found.
    """
    switch_name = module.params['switch_name']
    switch_info = module.client.api.get_device_by_name(switch_name)
    if not switch_info:
        module.fail_json(msg=str("Device with name '%s' does not exist." % switch_name))
    return switch_info