from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def are_attrs_equal(current_resource, module, attributes):
    """
    Check if the specified attributes are equal in the specified 'model' and 'module'. This is used to check if an OCI
    Model instance already has the values specified by an Ansible user while invoking an OCI Ansible module and if a
    resource needs to be updated.
    :param current_resource: A resource model instance
    :param module: The AnsibleModule representing the options provided by the user
    :param attributes: A list of attributes that would need to be compared in the model and the module instances.
    :return: True if the values for the list of attributes is the same in the model and module instances
    """
    for attr in attributes:
        curr_value = getattr(current_resource, attr, None)
        user_provided_value = _get_user_provided_value(module, attribute_name=attr)
        if user_provided_value is not None:
            if curr_value != user_provided_value:
                _debug("are_attrs_equal - current resource's attribute " + attr + ' value is ' + str(curr_value) + " and this doesn't match user provided value of " + str(user_provided_value))
                return False
    return True