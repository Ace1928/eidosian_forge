import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
@property
def pid(self):
    """ID of the process which created this tensor (an integer)."""
    return self._pid