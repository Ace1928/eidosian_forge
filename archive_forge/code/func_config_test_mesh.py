import multiprocessing
import os
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.python.platform import test as tf_test
def config_test_mesh(mesh: layout_lib.Mesh):
    """No Op.

  Args:
    mesh: The DTensor mesh.
  """
    if config.backend_is_pw():
        del mesh