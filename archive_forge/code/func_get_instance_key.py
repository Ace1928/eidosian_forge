import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def get_instance_key(self, group_key, device):
    """Returns a new instance key for use in defining a collective op.

    You should call this once per each collective op of a collective instance.

    Args:
      group_key: the group key returned by get_group_key(). You should not
        assign the group key yourself.
      device: a canonical device string. It should be the device this collective
        op is on.

    Returns:
      a new instance key.

    Raises:
      ValueError: when the group key is invalid or the device is not in the
      group.
    """
    with self._lock:
        group = self._instance_key_table.get(group_key, None)
        if group is None:
            raise ValueError(f'Group {group_key} is not found.')
        if device not in group:
            raise ValueError(f'Device {device} is not present in group {group_key}')
        v = group[device]
        group[device] += 1
        return v