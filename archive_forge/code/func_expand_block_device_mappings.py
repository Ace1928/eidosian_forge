from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_block_device_mappings(eg_launchspec, bdms):
    if bdms is not None:
        eg_bdms = []
        for bdm in bdms:
            eg_bdm = expand_fields(bdm_fields, bdm, 'BlockDeviceMapping')
            if bdm.get('ebs') is not None:
                eg_bdm.ebs = expand_fields(ebs_fields, bdm.get('ebs'), 'EBS')
            eg_bdms.append(eg_bdm)
        if len(eg_bdms) > 0:
            eg_launchspec.block_device_mappings = eg_bdms