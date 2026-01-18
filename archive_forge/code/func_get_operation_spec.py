from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def get_operation_spec(self, operation_name):
    if operation_name not in self._operation_spec_cache:
        self._operation_spec_cache[operation_name] = self._conn.get_operation_spec(operation_name)
    return self._operation_spec_cache[operation_name]