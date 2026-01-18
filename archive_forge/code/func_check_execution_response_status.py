from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def check_execution_response_status(self, response):
    """
        Checks the reponse status provided by API in the Cisco Catalyst Center

        Parameters:
            response (dict) - API response

        Returns:
            self
        """
    if not response:
        self.msg = 'response is empty'
        self.status = 'failed'
        return self
    if not isinstance(response, dict):
        self.msg = 'response is not a dictionary'
        self.status = 'failed'
        return self
    executionid = response.get('executionId')
    while True:
        execution_details = self.get_execution_details(executionid)
        if execution_details.get('status') == 'SUCCESS':
            self.result['changed'] = True
            self.msg = 'Successfully executed'
            self.status = 'success'
            break
        if execution_details.get('bapiError'):
            self.msg = execution_details.get('bapiError')
            self.status = 'failed'
            break
    return self