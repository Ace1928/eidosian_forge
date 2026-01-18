import abc
import math
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Text, Tuple, TypeVar, Union
from absl import logging
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
def get_list_of_hosts(strategy: tpu_strategy.TPUStrategy) -> List[Text]:
    """Returns a sorted list of CPU devices for the remote jobs.

  Args:
    strategy: A TPUStrategy object.

  Returns:
    A sorted list of device host strings.
  """
    list_of_hosts = []
    for tpu_device in _sort_device_spec_strings(strategy.extended.worker_devices):
        host = device_util.get_host_for_device(tpu_device)
        if host not in list_of_hosts:
            list_of_hosts.append(host)
    assert len(list_of_hosts) == strategy.extended.num_hosts
    return list_of_hosts