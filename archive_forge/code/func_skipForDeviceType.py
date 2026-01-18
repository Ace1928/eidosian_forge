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
def skipForDeviceType(self, device_type: typing.List[str], reason: str, unless_device_count_equals_to=None):
    """Skip the test for the specific device_type.

    Args:
      device_type: list of device types, one of "CPU", "GPU", or "TPU".
      reason: string that describe the reason for skipping the test.
      unless_device_count_equals_to: Optional int. This parameter only works if
        device_type is "TPU". If set, the test will be skipped unless the number
        of TPUs equals to the specified count.
    """
    physical_device_types = set([d.device_type for d in tf_config.list_physical_devices()])
    for device in device_type:
        if device == 'TPU' and is_tpu_present():
            if unless_device_count_equals_to is None:
                self.skipTest(reason)
            elif len(list_local_logical_devices(device)) != unless_device_count_equals_to:
                self.skipTest(reason)
        if device == 'CPU' and len(physical_device_types) == 1 and ('CPU' in physical_device_types):
            self.skipTest(reason)
        if device == 'GPU' and 'GPU' in physical_device_types:
            self.skipTest(reason)