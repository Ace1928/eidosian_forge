import os
import sys
import threading
from absl import logging
from keras.src.utils import keras_logging
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.utils.is_interactive_logging_enabled')
def is_interactive_logging_enabled():
    """Check if interactive logging is enabled.

    To switch between writing logs to stdout and `absl.logging`, you may use
    `keras.utils.enable_interactive_logging()` and
    `keras.utils.disable_interactive_logging()`.

    Returns:
      Boolean (True if interactive logging is enabled and False otherwise).
    """
    return getattr(INTERACTIVE_LOGGING, 'enable', keras_logging.INTERACTIVE_LOGGING_DEFAULT)