import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
@property
def num_bytes(self):
    """Size of this tensor in bytes (long integer)."""
    return self._num_bytes