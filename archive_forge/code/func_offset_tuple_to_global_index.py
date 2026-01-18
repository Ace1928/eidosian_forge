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
def offset_tuple_to_global_index(self, offset_tuple):
    """Mapping from offset to index in global tensor."""
    index = 0
    for i, o in enumerate(offset_tuple):
        m = 1
        for x in range(i + 1, self.rank):
            m = m * self.num_shards(x)
        index = index + m * o
    return index