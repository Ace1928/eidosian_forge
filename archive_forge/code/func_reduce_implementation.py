import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def reduce_implementation(self, reduce_op, per_replica_value, destinations, options):
    values_util.mark_as_unsaveable()
    all_reduced = self._all_reduce_per_replica_values(reduce_op, [per_replica_value], options)[0]
    devices = get_devices_from(destinations, self._canonicalize_devices)
    if _devices_match(per_replica_value, destinations, self._canonicalize_devices):
        return all_reduced
    if not isinstance(all_reduced, value_lib.Mirrored):
        all_reduced = value_lib.Mirrored([all_reduced])
    index = []
    with ops.control_dependencies(all_reduced.values):
        for d in devices:
            with ops.device(d):
                for v in all_reduced.values:
                    if v.device == d:
                        index.append(array_ops.identity(v))
                        break
                else:
                    index.append(array_ops.identity(all_reduced._primary))
    return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)