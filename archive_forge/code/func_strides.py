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
@property
def strides(self) -> List[int]:
    """Returns the strides tensor array for this mesh.

    If the mesh shape is `[a, b, c, d]`, then the strides array can be computed
    as `[b*c*d, c*d, d, 1]`. This array can be useful in computing local device
    offsets given a device ID. Using the same example, the device coordinates of
    the mesh can be computed as:

    ```
    [(device_id / (b*c*d)) % a,
     (device_id / (c*d))   % b,
     (device_id / (d))     % c,
     (device_id)           % d]
    ```

    This is the same as `(device_id // mesh.strides) % mesh.shape`.

    Returns:
      The mesh strides as an integer tensor.
    """
    return _compute_mesh_strides(self.shape())