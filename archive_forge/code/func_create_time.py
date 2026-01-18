import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
@property
def create_time(self):
    """Timestamp when this tensor was created (long integer)."""
    return self._create_time