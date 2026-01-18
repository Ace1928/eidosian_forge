import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
@property
def object_id(self):
    """Returns the object identifier of this tensor (integer)."""
    return self._object_id