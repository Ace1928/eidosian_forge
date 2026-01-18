from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_device_uuids(self, site_name, device_family, device_role, device_series_name=None):
    """
        Retrieve a list of device UUIDs based on the specified criteria.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            site_name (str): The name of the site for which device UUIDs are requested.
            device_family (str): The family/type of devices to filter on.
            device_role (str): The role of devices to filter on. If None, 'ALL' roles are considered.
            device_series_name(str): Specifies the name of the device series.
        Returns:
            list: A list of device UUIDs that match the specified criteria.
        Description:
            The function checks the reachability status and role of devices in the given site.
            Only devices with "Reachable" status are considered, and filtering is based on the specified
            device family and role (if provided).
        """
    device_uuid_list = []
    if not site_name:
        site_name = 'Global'
        self.log("Since site name is not given so it will be fetch all the devices under Global and mark site name as 'Global'", 'INFO')
    site_exists, site_id = self.site_exists(site_name)
    if not site_exists:
        self.log("Site '{0}' is not found in the Cisco Catalyst Center, hence unable to fetch associated\n                        devices.".format(site_name), 'INFO')
        return device_uuid_list
    if device_series_name:
        if device_series_name.startswith('.*') and device_series_name.endswith('.*'):
            self.log("Device series name '{0}' is already in the regex format".format(device_series_name), 'INFO')
        else:
            device_series_name = '.*' + device_series_name + '.*'
    site_params = {'site_id': site_id, 'device_family': device_family}
    response = self.dnac._exec(family='sites', function='get_membership', op_modifies=True, params=site_params)
    self.log("Received API response from 'get_membership': {0}".format(str(response)), 'DEBUG')
    response = response['device']
    site_response_list = []
    for item in response:
        if item['response']:
            for item_dict in item['response']:
                site_response_list.append(item_dict)
    if device_role.upper() == 'ALL':
        device_role = None
    device_params = {'series': device_series_name, 'family': device_family, 'role': device_role}
    device_list_response = self.dnac._exec(family='devices', function='get_device_list', op_modifies=True, params=device_params)
    device_response = device_list_response.get('response')
    if not response or not device_response:
        self.log("Failed to retrieve devices associated with the site '{0}' due to empty API response.".format(site_name), 'INFO')
        return device_uuid_list
    site_memberships_ids, device_response_ids = ([], [])
    for item in site_response_list:
        if item['reachabilityStatus'] != 'Reachable':
            self.log("Device '{0}' is currently '{1}' and cannot be included in the SWIM distribution/activation\n                            process.".format(item['managementIpAddress'], item['reachabilityStatus']), 'INFO')
            continue
        self.log("Device '{0}' from site '{1}' is ready for the SWIM distribution/activation\n                        process.".format(item['managementIpAddress'], site_name), 'INFO')
        site_memberships_ids.append(item['instanceUuid'])
    for item in device_response:
        if item['reachabilityStatus'] != 'Reachable':
            self.log("Unable to proceed with the device '{0}' for SWIM distribution/activation as its status is\n                            '{1}'.".format(item['managementIpAddress'], item['reachabilityStatus']), 'INFO')
            continue
        self.log("Device '{0}' matches to the specified filter requirements and is set for SWIM\n                      distribution/activation.".format(item['managementIpAddress']), 'INFO')
        device_response_ids.append(item['instanceUuid'])
    device_uuid_list = set(site_memberships_ids).intersection(set(device_response_ids))
    return device_uuid_list