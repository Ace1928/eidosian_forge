import os
import sys
import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def has_verbosity(level):
    return get_verbosity() >= level