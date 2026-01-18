import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def is_registered(self, prefix):
    """Test if a command prefix or its alias is has a registered handler.

    Args:
      prefix: A prefix or its alias, as a str.

    Returns:
      True iff a handler is registered for prefix.
    """
    return self._resolve_prefix(prefix) is not None