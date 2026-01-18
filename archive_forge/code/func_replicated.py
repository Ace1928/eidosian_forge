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
def replicated(cls, mesh: Mesh, rank: int) -> 'Layout':
    """Returns a replicated layout of rank `rank`."""
    return cls._new_object(mesh=mesh, rank=rank)