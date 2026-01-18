from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_diff_tagging(self):
    """
        Tag or untag a software image as golden based on provided tagging details.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            This function tags or untags a software image as a golden image in Cisco Catalyst Center based on the provided
            tagging details. The tagging action is determined by the value of the 'tagging' attribute
            in the 'tagging_details' dictionary.If 'tagging' is True, the image is tagged as golden, and if 'tagging'
            is False, the golden tag is removed. The function sends the appropriate request to Cisco Catalyst Center and updates the
            task details in the 'result' dictionary. If the operation is successful, 'changed' is set to True.
        """
    tagging_details = self.want.get('tagging_details')
    tag_image_golden = tagging_details.get('tagging')
    image_name = self.get_image_name_from_id(self.have.get('tagging_image_id'))
    image_params = dict(image_id=self.have.get('tagging_image_id'), site_id=self.have.get('site_id'), device_family_identifier=self.have.get('device_family_identifier'), device_role=tagging_details.get('device_role', 'ALL').upper())
    response = self.dnac._exec(family='software_image_management_swim', function='get_golden_tag_status_of_an_image', op_modifies=True, params=image_params)
    self.log("Received API response from 'get_golden_tag_status_of_an_image': {0}".format(str(response)), 'DEBUG')
    response = response.get('response')
    if response:
        image_status = response['taggedGolden']
        if image_status and image_status == tag_image_golden:
            self.status = 'success'
            self.result['changed'] = False
            self.msg = "SWIM Image '{0}' already tagged as Golden image in Cisco Catalyst Center".format(image_name)
            self.result['msg'] = self.msg
            self.log(self.msg, 'INFO')
            return self
        if not image_status and image_status == tag_image_golden:
            self.status = 'success'
            self.result['changed'] = False
            self.msg = "SWIM Image '{0}' already un-tagged from Golden image in Cisco Catalyst Center".format(image_name)
            self.result['msg'] = self.msg
            self.log(self.msg, 'INFO')
            return self
    if tag_image_golden:
        image_params = dict(imageId=self.have.get('tagging_image_id'), siteId=self.have.get('site_id'), deviceFamilyIdentifier=self.have.get('device_family_identifier'), deviceRole=tagging_details.get('device_role', 'ALL').upper())
        self.log('Parameters for tagging the image as golden: {0}'.format(str(image_params)), 'INFO')
        response = self.dnac._exec(family='software_image_management_swim', function='tag_as_golden_image', op_modifies=True, params=image_params)
        self.log("Received API response from 'tag_as_golden_image': {0}".format(str(response)), 'DEBUG')
    else:
        self.log('Parameters for un-tagging the image as golden: {0}'.format(str(image_params)), 'INFO')
        response = self.dnac._exec(family='software_image_management_swim', function='remove_golden_tag_for_image', op_modifies=True, params=image_params)
        self.log("Received API response from 'remove_golden_tag_for_image': {0}".format(str(response)), 'DEBUG')
    if not response:
        self.status = 'failed'
        self.msg = 'Did not get the response of API so cannot check the Golden tagging status of image - {0}'.format(image_name)
        self.log(self.msg, 'ERROR')
        self.result['response'] = self.msg
        return self
    task_details = {}
    task_id = response.get('response').get('taskId')
    while True:
        task_details = self.get_task_details(task_id)
        if not task_details.get('isError') and 'successful' in task_details.get('progress'):
            self.status = 'success'
            self.result['changed'] = True
            self.msg = task_details.get('progress')
            self.result['msg'] = self.msg
            self.result['response'] = self.msg
            self.log(self.msg, 'INFO')
            break
        elif task_details.get('isError'):
            failure_reason = task_details.get('failureReason', '')
            if failure_reason and 'An inheritted tag cannot be un-tagged' in failure_reason:
                self.status = 'failed'
                self.result['changed'] = False
                self.msg = failure_reason
                self.result['msg'] = failure_reason
                self.log(self.msg, 'ERROR')
                self.result['response'] = self.msg
                break
            else:
                error_message = task_details.get('failureReason', 'Error: while tagging/un-tagging the golden swim image.')
                self.status = 'failed'
                self.msg = error_message
                self.result['msg'] = error_message
                self.log(self.msg, 'ERROR')
                self.result['response'] = self.msg
                break
    return self