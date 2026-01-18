import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def set_logical_devices_to_at_least(device, num):
    """Create logical devices of at least a given number."""
    if num < 1:
        raise ValueError('`num` must be at least 1 not %r' % (num,))
    physical_devices = config.list_physical_devices(device)
    if not physical_devices:
        raise RuntimeError('No {} found'.format(device))
    if len(physical_devices) >= num:
        return
    num = num - len(physical_devices) + 1
    logical_devices = []
    for _ in range(num):
        if device.upper() == 'GPU':
            logical_devices.append(context.LogicalDeviceConfiguration(memory_limit=2048))
        else:
            logical_devices.append(context.LogicalDeviceConfiguration())
    config.set_logical_device_configuration(physical_devices[-1], logical_devices)