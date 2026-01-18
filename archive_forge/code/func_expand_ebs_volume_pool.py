from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_ebs_volume_pool(eg_compute, ebs_volumes_list):
    if ebs_volumes_list is not None:
        eg_volumes = []
        for volume in ebs_volumes_list:
            eg_volume = spotinst.aws_elastigroup.EbsVolume()
            if volume.get('device_name') is not None:
                eg_volume.device_name = volume.get('device_name')
            if volume.get('volume_ids') is not None:
                eg_volume.volume_ids = volume.get('volume_ids')
            if eg_volume.device_name is not None:
                eg_volumes.append(eg_volume)
        if len(eg_volumes) > 0:
            eg_compute.ebs_volume_pool = eg_volumes