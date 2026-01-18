from __future__ import absolute_import, division, print_function
import json
import os
import sys
from ansible_collections.ansible.netcommon.plugins.sub_plugins.grpc.base import (
@ensure_connect
def replace_config(self, path):
    """Replace grpc call equivalent  of PATCH RESTconf call
        :param data: JSON
        :type data: str
        :return: Return the response object
        :rtype: Response object
        """
    path = json.dumps(path)
    stub = self._ems_grpc_pb2.beta_create_gRPCConfigOper_stub(self._connection._channel)
    message = self._ems_grpc_pb2.ConfigArgs(yangjson=path)
    response = stub.ReplaceConfig(message, self._connection._timeout, metadata=self._connection._login_credentials)
    if response:
        return response.errors
    else:
        return None