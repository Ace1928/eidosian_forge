from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def check_for_sdc_modification(volume, sdc_id, sdc_details):
    """
    :param volume: The volume details
    :param sdc_id: The ID of the SDC
    :param sdc_details: The details of SDC
    :return: Dictionary with SDC attributes to be modified
    """
    access_mode_dict = dict()
    limits_dict = dict()
    for sdc in volume['mappedSdcInfo']:
        if sdc['sdcId'] == sdc_id:
            if sdc['accessMode'] != get_access_mode(sdc_details['access_mode']):
                access_mode_dict['sdc_id'] = sdc_id
                access_mode_dict['accessMode'] = get_access_mode(sdc_details['access_mode'])
            if sdc['limitIops'] != sdc_details['iops_limit'] or sdc['limitBwInMbps'] != sdc_details['bandwidth_limit']:
                limits_dict['sdc_id'] = sdc_id
                limits_dict['iops_limit'] = None
                limits_dict['bandwidth_limit'] = None
                if sdc['limitIops'] != sdc_details['iops_limit']:
                    limits_dict['iops_limit'] = sdc_details['iops_limit']
                if sdc['limitBwInMbps'] != get_limits_in_mb(sdc_details['bandwidth_limit']):
                    limits_dict['bandwidth_limit'] = sdc_details['bandwidth_limit']
            break
    return (access_mode_dict, limits_dict)