from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def crud_operation(self, op_name, params):
    """
        Allow user request execution of simple operations(natively supported by API provider) only.

        :param op_name: name of the operation being called by the user
        :type op_name: str
        :param params: definition of the params that operation should be executed with
        :type params: dict
        :return: Result of the operation being executed
        :rtype: dict
        """
    op_spec = self.get_operation_spec(op_name)
    if op_spec is None:
        raise FtdInvalidOperationNameError(op_name)
    if self._operation_checker.is_add_operation(op_name, op_spec):
        resp = self.add_object(op_name, params)
    elif self._operation_checker.is_edit_operation(op_name, op_spec):
        resp = self.edit_object(op_name, params)
    elif self._operation_checker.is_delete_operation(op_name, op_spec):
        resp = self.delete_object(op_name, params)
    elif self._operation_checker.is_find_by_filter_operation(op_name, params, op_spec):
        resp = list(self.get_objects_by_filter(op_name, params))
    else:
        resp = self.send_general_request(op_name, params)
    return resp