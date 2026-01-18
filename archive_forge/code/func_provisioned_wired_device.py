from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def provisioned_wired_device(self):
    """
        Provision wired devices in Cisco Catalyst Center.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of the class with updated result, status, and log.
        Description:
            This function provisions wired devices in Cisco Catalyst Center based on the configuration provided.
            It retrieves the site name and IP addresses of the devices from the list of configuration,
            attempts to provision each device with site, and monitors the provisioning process.
        """
    provision_wired_list = self.config[0]['provision_wired_device']
    total_devices_to_provisioned = len(provision_wired_list)
    device_ip_list = []
    provision_count, already_provision_count = (0, 0)
    for prov_dict in provision_wired_list:
        managed_flag = False
        device_ip = prov_dict['device_ip']
        device_ip_list.append(device_ip)
        site_name = prov_dict['site_name']
        device_type = 'Wired'
        resync_retry_count = prov_dict.get('resync_retry_count', 200)
        resync_retry_interval = prov_dict.get('resync_retry_interval', 2)
        if not site_name or not device_ip:
            self.status = 'failed'
            self.msg = 'Site and Device IP are required for Provisioning of Wired Devices.'
            self.log(self.msg, 'ERROR')
            self.result['response'] = self.msg
            return self
        provision_wired_params = {'deviceManagementIpAddress': device_ip, 'siteNameHierarchy': site_name}
        while resync_retry_count:
            response = self.get_device_response(device_ip)
            self.log('Device is in {0} state waiting for Managed State.'.format(response['managementState']), 'DEBUG')
            if response.get('managementState') == 'Managed' and response.get('collectionStatus') == 'Managed' and response.get('hostname'):
                msg = "Device '{0}' comes to managed state and ready for provisioning with the resync_retry_count\n                        '{1}' left having resync interval of {2} seconds".format(device_ip, resync_retry_count, resync_retry_interval)
                self.log(msg, 'INFO')
                managed_flag = True
                break
            if response.get('collectionStatus') == 'Partial Collection Failure' or response.get('collectionStatus') == 'Could Not Synchronize':
                device_status = response.get('collectionStatus')
                msg = "Device '{0}' comes to '{1}' state and never goes for provisioning with the resync_retry_count\n                        '{2}' left having resync interval of {3} seconds".format(device_ip, device_status, resync_retry_count, resync_retry_interval)
                self.log(msg, 'INFO')
                managed_flag = False
                break
            time.sleep(resync_retry_interval)
            resync_retry_count = resync_retry_count - 1
        if not managed_flag:
            self.log('Device {0} is not transitioning to the managed state, so provisioning operation cannot\n                            be performed.'.format(device_ip), 'WARNING')
            continue
        try:
            response = self.dnac._exec(family='sda', function='provision_wired_device', op_modifies=True, params=provision_wired_params)
            if response.get('status') == 'failed':
                description = response.get('description')
                error_msg = 'Cannot do Provisioning for device {0} beacuse of {1}'.format(device_ip, description)
                self.log(error_msg)
                continue
            task_id = response.get('taskId')
            while True:
                execution_details = self.get_task_details(task_id)
                progress = execution_details.get('progress')
                if 'TASK_PROVISION' in progress:
                    self.handle_successful_provisioning(device_ip, execution_details, device_type)
                    provision_count += 1
                    break
                elif execution_details.get('isError'):
                    self.handle_failed_provisioning(device_ip, execution_details, device_type)
                    break
        except Exception as e:
            self.handle_provisioning_exception(device_ip, e, device_type)
            if 'already provisioned' in str(e):
                self.log(str(e), 'INFO')
                already_provision_count += 1
    if already_provision_count == total_devices_to_provisioned:
        self.handle_all_already_provisioned(device_ip_list, device_type)
    elif provision_count == total_devices_to_provisioned:
        self.handle_all_provisioned(device_type)
    elif provision_count == 0:
        self.handle_all_failed_provision(device_type)
    else:
        self.handle_partially_provisioned(provision_count, device_type)
    return self