import collections
import functools
import itertools
from typing import List, Dict, Optional, Union
import numpy as np
from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export
@classmethod
def inner_sharded(cls, mesh: Mesh, inner_dim: str, rank: int) -> 'Layout':
    """Returns a layout sharded on inner dimension."""
    return cls.batch_sharded(mesh, inner_dim, rank, axis=rank - 1)