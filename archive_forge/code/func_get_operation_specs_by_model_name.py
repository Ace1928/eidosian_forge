from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def get_operation_specs_by_model_name(self, model_name):
    if model_name not in self._models_operations_specs_cache:
        model_op_specs = self._conn.get_operation_specs_by_model_name(model_name)
        self._models_operations_specs_cache[model_name] = model_op_specs
        for op_name, op_spec in iteritems(model_op_specs):
            self._operation_spec_cache.setdefault(op_name, op_spec)
    return self._models_operations_specs_cache[model_name]