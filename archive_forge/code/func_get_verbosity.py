import os
import sys
import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def get_verbosity():
    global verbosity_level
    if verbosity_level is not None:
        return verbosity_level
    return int(os.getenv(VERBOSITY_VAR_NAME, DEFAULT_VERBOSITY))