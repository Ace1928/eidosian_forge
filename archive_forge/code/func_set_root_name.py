import re
from tensorflow.python.util import tf_inspect
def set_root_name(self, root_name):
    """Override the default root name of 'tf'."""
    self._root_name = root_name