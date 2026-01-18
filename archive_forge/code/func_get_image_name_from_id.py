from __future__ import absolute_import, division, print_function
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
from ansible.module_utils.basic import AnsibleModule
import os
import time
def get_image_name_from_id(self, image_id):
    """
        Retrieve the unique image name based on the provided image id.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            id (str): The unique image ID (UUID) of the software image to search for.
        Returns:
            str: The image name corresponding to the given unique image ID (UUID)
        Raises:
            AnsibleFailJson: If the image is not found in the response.
        Description:
            This function sends a request to Cisco Catalyst Center to retrieve details about a software image based on its id.
            It extracts and returns the image name if a single matching image is found. If no image or multiple
            images are found with the same name, it raises an exception.
        """
    image_response = self.dnac._exec(family='software_image_management_swim', function='get_software_image_details', params={'image_uuid': image_id})
    self.log("Received API response from 'get_software_image_details': {0}".format(str(image_response)), 'DEBUG')
    image_list = image_response.get('response')
    if len(image_list) == 1:
        image_name = image_list[0].get('name')
        self.log("SWIM image '{0}' has been fetched successfully from Cisco Catalyst Center".format(image_name), 'INFO')
    else:
        error_message = "SWIM image with Id '{0}' could not be found in Cisco Catalyst Center".format(image_id)
        self.log(error_message, 'ERROR')
        self.module.fail_json(msg=error_message, response=image_response)
    return image_name