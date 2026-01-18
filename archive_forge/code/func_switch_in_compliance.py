from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def switch_in_compliance(module, sw_info):
    """ Check if switch is currently in compliance.

    :param module: Ansible module with parameters and client connection.
    :param sw_info: Dict of switch info.
    :return: Nothing or exit with failure if device is not in compliance.
    """
    compliance = module.client.api.check_compliance(sw_info['key'], sw_info['type'])
    if compliance['complianceCode'] != '0000':
        module.fail_json(msg=str('Switch %s is not in compliance. Returned compliance code %s.' % (sw_info['fqdn'], compliance['complianceCode'])))