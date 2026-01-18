import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def get_op_def(self, op_name):
    if op_name in self._op_per_name:
        return self._op_per_name[op_name]
    raise ValueError(f'No op_def found for op name {op_name}.')