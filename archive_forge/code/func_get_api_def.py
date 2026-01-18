import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def get_api_def(self, op_name):
    api_def_proto = api_def_pb2.ApiDef()
    buf = c_api.TF_ApiDefMapGet(self._api_def_map, op_name, len(op_name))
    try:
        api_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
    finally:
        c_api.TF_DeleteBuffer(buf)
    return api_def_proto