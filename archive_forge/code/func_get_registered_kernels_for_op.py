from tensorflow.core.framework import kernel_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
def get_registered_kernels_for_op(name):
    """Returns a KernelList proto of registered kernels for a given op.

  Args:
    name: A string representing the name of the op whose kernels to retrieve.
  """
    buf = c_api.TF_GetRegisteredKernelsForOp(name)
    data = c_api.TF_GetBuffer(buf)
    kernel_list = kernel_def_pb2.KernelList()
    kernel_list.ParseFromString(compat.as_bytes(data))
    return kernel_list