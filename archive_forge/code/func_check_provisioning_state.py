from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def check_provisioning_state(self, azure_object, requested_state='present'):
    """
        Check an Azure object's provisioning state. If something did not complete the provisioning
        process, then we cannot operate on it.

        :param azure_object An object such as a subnet, storageaccount, etc. Must have provisioning_state
                            and name attributes.
        :return None
        """
    if hasattr(azure_object, 'properties') and hasattr(azure_object.properties, 'provisioning_state') and hasattr(azure_object, 'name'):
        if isinstance(azure_object.properties.provisioning_state, Enum):
            if azure_object.properties.provisioning_state.value != AZURE_SUCCESS_STATE and requested_state != 'absent':
                self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.properties.provisioning_state, AZURE_SUCCESS_STATE))
            return
        if azure_object.properties.provisioning_state != AZURE_SUCCESS_STATE and requested_state != 'absent':
            self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.properties.provisioning_state, AZURE_SUCCESS_STATE))
        return
    if hasattr(azure_object, 'provisioning_state') or not hasattr(azure_object, 'name'):
        if isinstance(azure_object.provisioning_state, Enum):
            if azure_object.provisioning_state.value != AZURE_SUCCESS_STATE and requested_state != 'absent':
                self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.provisioning_state, AZURE_SUCCESS_STATE))
            return
        if azure_object.provisioning_state != AZURE_SUCCESS_STATE and requested_state != 'absent':
            self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.provisioning_state, AZURE_SUCCESS_STATE))