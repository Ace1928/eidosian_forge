from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_diff_activation(self):
    """
        Get image activation parameters from the playbook and trigger image activation.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This function retrieves image activation parameters from the playbook's 'activation_details' and triggers the
            activation of the specified software image on the specified device. It monitors the activation task's progress and
            updates the 'result' dictionary. If the operation is successful, 'changed' is set to True.
        """
    activation_details = self.want.get('activation_details')
    site_name = activation_details.get('site_name')
    device_family = activation_details.get('device_family_name')
    device_role = activation_details.get('device_role', 'ALL')
    device_series_name = activation_details.get('device_series_name')
    device_uuid_list = self.get_device_uuids(site_name, device_family, device_role, device_series_name)
    image_id = self.have.get('activation_image_id')
    self.complete_successful_activation = False
    self.partial_successful_activation = False
    self.single_device_activation = False
    if self.have.get('activation_device_id'):
        payload = [dict(activateLowerImageVersion=activation_details.get('activate_lower_image_version'), deviceUpgradeMode=activation_details.get('device_upgrade_mode'), distributeIfNeeded=activation_details.get('distribute_if_needed'), deviceUuid=self.have.get('activation_device_id'), imageUuidList=[image_id])]
        activation_params = dict(schedule_validate=activation_details.get('scehdule_validate'), payload=payload)
        self.log('Activation Params: {0}'.format(str(activation_params)), 'INFO')
        response = self.dnac._exec(family='software_image_management_swim', function='trigger_software_image_activation', op_modifies=True, params=activation_params)
        self.log("Received API response from 'trigger_software_image_activation': {0}".format(str(response)), 'DEBUG')
        task_details = {}
        task_id = response.get('response').get('taskId')
        while True:
            task_details = self.get_task_details(task_id)
            if not task_details.get('isError') and 'completed successfully' in task_details.get('progress'):
                self.result['changed'] = True
                self.result['msg'] = 'Image Activated successfully'
                self.status = 'success'
                self.single_device_activation = True
                break
            if task_details.get('isError'):
                self.msg = "Activation for Image with Id '{0}' gets failed".format(image_id)
                self.status = 'failed'
                self.result['response'] = task_details
                self.log(self.msg, 'ERROR')
                return self
        self.result['response'] = task_details if task_details else response
        return self
    if len(device_uuid_list) == 0:
        self.status = 'success'
        self.msg = 'The SWIM image activation task could not proceed because no eligible devices were found.'
        self.result['msg'] = self.msg
        self.log(self.msg, 'WARNING')
        return self
    self.log('Device UUIDs involved in Image Activation: {0}'.format(str(device_uuid_list)), 'INFO')
    activation_task_dict = {}
    for device_uuid in device_uuid_list:
        device_management_ip = self.get_device_ip_from_id(device_uuid)
        payload = [dict(activateLowerImageVersion=activation_details.get('activate_lower_image_version'), deviceUpgradeMode=activation_details.get('device_upgrade_mode'), distributeIfNeeded=activation_details.get('distribute_if_needed'), deviceUuid=device_uuid, imageUuidList=[image_id])]
        activation_params = dict(schedule_validate=activation_details.get('scehdule_validate'), payload=payload)
        self.log('Activation Params: {0}'.format(str(activation_params)), 'INFO')
        response = self.dnac._exec(family='software_image_management_swim', function='trigger_software_image_activation', op_modifies=True, params=activation_params)
        self.log("Received API response from 'trigger_software_image_activation': {0}".format(str(response)), 'DEBUG')
        if response:
            task_details = {}
            task_id = response.get('response').get('taskId')
            activation_task_dict[device_management_ip] = task_id
    device_ips_list, device_activation_count = self.check_swim_task_status(activation_task_dict, 'Activation')
    if device_activation_count == 0:
        self.status = 'failed'
        self.msg = "Image with Id '{0}' activation failed for all devices".format(image_id)
    elif device_activation_count == len(device_uuid_list):
        self.result['changed'] = True
        self.status = 'success'
        self.complete_successful_activation = True
        self.msg = "Image with Id '{0}' activated successfully for all devices".format(image_id)
    else:
        self.result['changed'] = True
        self.status = 'success'
        self.partial_successful_activation = True
        self.msg = "Image with Id '{0}' activated and partially successfull".format(image_id)
        self.log('For Device(s) {0} Image activation gets Failed'.format(str(device_ips_list)), 'CRITICAL')
    self.result['msg'] = self.msg
    self.log(self.msg, 'INFO')
    return self