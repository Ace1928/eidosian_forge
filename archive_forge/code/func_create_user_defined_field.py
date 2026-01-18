from __future__ import absolute_import, division, print_function
import csv
import time
from datetime import datetime
from io import BytesIO, StringIO
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def create_user_defined_field(self, udf):
    """
        Create a Global User Defined Field in Cisco Catalyst Center based on the provided configuration.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            udf (dict): A dictionary having the payload for the creation of user defined field(UDF) in Cisco Catalyst Center.
        Returns:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Description:
            The function retrieves the configuration for adding a user-defined field from the configuration object,
            sends the request to Cisco Catalyst Center to create the field, and logs the response.
        """
    try:
        response = self.dnac._exec(family='devices', function='create_user_defined_field', params=udf)
        self.log("Received API response from 'create_user_defined_field': {0}".format(str(response)), 'DEBUG')
        response = response.get('response')
        field_name = udf.get('name')
        self.log("Global User Defined Field with name '{0}' created successfully".format(field_name), 'INFO')
        self.status = 'success'
    except Exception as e:
        error_message = 'Error while creating Global UDF(User Defined Field) in Cisco Catalyst Center: {0}'.format(str(e))
        self.log(error_message, 'ERROR')
    return self