import threading
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def is_cleared(self):
    return not self.stack