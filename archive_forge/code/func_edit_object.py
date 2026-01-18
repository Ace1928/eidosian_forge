from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def edit_object(self, operation_name, params):
    data, dummy, path_params = _get_user_params(params)
    model_name = self.get_operation_spec(operation_name)[OperationField.MODEL_NAME]
    get_operation = self._find_get_operation(model_name)
    if get_operation:
        existing_object = self.send_general_request(get_operation, {ParamName.PATH_PARAMS: path_params})
        if not existing_object:
            raise FtdConfigurationError('Referenced object does not exist')
        elif equal_objects(existing_object, data):
            return existing_object
    return self.send_general_request(operation_name, params)