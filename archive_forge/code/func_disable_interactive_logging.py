import os
import sys
import threading
from absl import logging
from keras.src.utils import keras_logging
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.utils.disable_interactive_logging')
def disable_interactive_logging():
    """Turn off interactive logging.

    When interactive logging is disabled, Keras sends logs to `absl.logging`.
    This is the best option when using Keras in a non-interactive
    way, such as running a training or inference job on a server.
    """
    INTERACTIVE_LOGGING.enable = False