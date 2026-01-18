import os
import sys
import threading
from absl import logging
from keras.src.utils import keras_logging
from tensorflow.python.util.tf_export import keras_export
@logging.skip_log_prefix
def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    if is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + '\n')
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)