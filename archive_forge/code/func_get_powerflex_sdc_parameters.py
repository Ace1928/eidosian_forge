from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_powerflex_sdc_parameters():
    """This method provide parameter required for the Ansible SDC module on
    PowerFlex"""
    return dict(sdc_id=dict(), sdc_ip=dict(), sdc_name=dict(), sdc_new_name=dict(), performance_profile=dict(choices=['Compact', 'HighPerformance']), state=dict(required=True, type='str', choices=['present', 'absent']))