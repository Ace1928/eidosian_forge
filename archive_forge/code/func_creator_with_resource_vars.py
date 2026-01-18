import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def creator_with_resource_vars(next_creator, **kwargs):
    """Variable creator to use in `_CurrentDistributionContext`."""
    if ops.inside_function():
        if_graph_building = 'graph_building'
    else:
        if_graph_building = 'not_graph_building'
    with monitoring.MonitoredTimer(distributed_variable_creation_time_counter.get_cell(strategy.__class__.__name__, if_graph_building)):
        _require_strategy_scope_extended(self)
        kwargs['use_resource'] = True
        kwargs['distribute_strategy'] = strategy
        if isinstance(kwargs['initial_value'], trackable.CheckpointInitialValue):
            checkpoint_restore_uid = kwargs['initial_value'].checkpoint_position.restore_uid
            kwargs['initial_value'] = kwargs['initial_value'].wrapped_value
        elif isinstance(kwargs['initial_value'], trackable.CheckpointInitialValueCallable):
            checkpoint_restore_uid = kwargs['initial_value'].checkpoint_position.restore_uid
        elif isinstance(kwargs['initial_value'], functools.partial) and isinstance(kwargs['initial_value'].func, trackable.CheckpointInitialValueCallable):
            checkpoint_restore_uid = kwargs['initial_value'].func.checkpoint_position.restore_uid
        else:
            checkpoint_restore_uid = None
        created = self._create_variable(next_creator, **kwargs)
        if checkpoint_restore_uid is not None:
            created._maybe_initialize_trackable()
            created._update_uid = checkpoint_restore_uid
        return created