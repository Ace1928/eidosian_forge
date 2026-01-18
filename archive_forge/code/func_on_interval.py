import abc
import threading
from absl import logging
from tensorflow.python.util.tf_export import keras_export
@abc.abstractmethod
def on_interval(self):
    """User-defined behavior that is called in the thread."""
    raise NotImplementedError('Runs every x interval seconds. Needs to be implemented in subclasses of `TimedThread`')