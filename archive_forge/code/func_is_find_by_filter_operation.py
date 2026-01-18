from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
@classmethod
def is_find_by_filter_operation(cls, operation_name, params, operation_spec):
    """
        Checks whether the called operation is 'find by filter'. This operation fetches all objects and finds
        the matching ones by the given filter. As filtering is done on the client side, this operation should be used
        only when selected filters are not implemented on the server side.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :param params: params - params should contain 'filters'
        :return: True if the called operation is find by filter, otherwise False
        :rtype: bool
        """
    is_get_list = cls.is_get_list_operation(operation_name, operation_spec)
    return is_get_list and ParamName.FILTERS in params and params[ParamName.FILTERS]