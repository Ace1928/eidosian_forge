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
def from_single_device_mesh(cls, mesh: Mesh) -> 'Layout':
    """Constructs a single device layout from a single device mesh."""
    return cls._new_object(mesh=mesh)