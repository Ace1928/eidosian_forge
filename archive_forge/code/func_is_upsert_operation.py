from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
@classmethod
def is_upsert_operation(cls, operation_name):
    """
        Check if operation defined with 'operation_name' is upsert objects operation according to 'operation_name'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :return: True if the called operation is upsert object operation, otherwise False
        :rtype: bool
        """
    return operation_name.startswith(OperationNamePrefix.UPSERT)