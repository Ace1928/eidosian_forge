import multiprocessing
import os
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.python.platform import test as tf_test
def slice_host_devices_for_multiworker(num_clients, client_id, ports):
    """Configure the current process to only use a slice of devices."""
    if num_clients == 0:
        del os.environ['CUDA_VISIBLE_DEVICES']
        del os.environ['HIP_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{client_id}'
        os.environ['HIP_VISIBLE_DEVICES'] = f'{client_id}'
        os.environ['CLOUD_TPU_TASK_ID'] = f'{client_id}'
        if 'tpu' in DTENSOR_TEST_UTIL_BACKEND.value:
            del ports
            raise NotImplementedError('OSS multi-client tests of TPU is not supported.')