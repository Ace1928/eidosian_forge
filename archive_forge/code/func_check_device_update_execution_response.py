from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def check_device_update_execution_response(self, response, device_ip):
    """
        Check the execution response of a device update task.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            response (dict): The response received after initiating the device update task.
            device_ip (str): The IP address of the device for which the update is performed.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This method checks the execution response of a device update task in Cisco Catalyst Center.
            It continuously queries the task details until the task is completed or an error occurs.
            If the task is successful, it sets the status to 'success' and logs an informational message.
            If the task fails, it sets the status to 'failed' and logs an error message with the failure reason, if available.
        """
    task_id = response.get('response').get('taskId')
    while True:
        execution_details = self.get_task_details(task_id)
        if execution_details.get('isError'):
            self.status = 'failed'
            failure_reason = execution_details.get('failureReason')
            if failure_reason:
                self.msg = "Device Updation for device '{0}' get failed due to {1}".format(device_ip, failure_reason)
            else:
                self.msg = "Device Updation for device '{0}' get failed".format(device_ip)
            self.log(self.msg, 'ERROR')
            break
        elif execution_details.get('endTime'):
            self.status = 'success'
            self.result['changed'] = True
            self.result['response'] = execution_details
            self.msg = "Device '{0}' present in Cisco Catalyst Center and have been updated successfully".format(device_ip)
            self.log(self.msg, 'INFO')
            break
    return self