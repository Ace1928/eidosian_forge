import collections
import copy
import itertools
import json
import os
import typing
from absl import flags
from absl.testing import parameterized
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.config import is_gpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import is_tpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import preferred_device_type  # pylint: disable=unused-import
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.dtensor.python.tests.test_backend_name import DTensorTestUtilBackend
from tensorflow.dtensor.python.tests.test_backend_util import DTensorTestBackendConfigurator
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test as tf_test
def is_local(device):
    spec = tf_device.DeviceSpec.from_string(device.name)
    if spec.job is None or spec.job == 'localhost':
        return True
    elif spec.job == config.job_name() and spec.task == config.client_id():
        return True
    return False