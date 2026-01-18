from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
@classmethod
def is_edit_operation(cls, operation_name, operation_spec):
    """
        Check if operation defined with 'operation_name' is edit object operation according to 'operation_spec'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :return: True if the called operation is edit object operation, otherwise False
        :rtype: bool
        """
    return operation_name.startswith(OperationNamePrefix.EDIT) and is_put_request(operation_spec)