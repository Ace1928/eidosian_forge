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
def local_device_locations(self) -> List[Dict[str, int]]:
    """Returns a list of local device locations.

    A device location is a dictionary from dimension names to indices on those
    dimensions.
    """
    mapping = self.unravel_index()
    return [mapping[device_id] for device_id in self.local_device_ids()]