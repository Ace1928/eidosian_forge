import collections
import re
from tensorflow.python.util import tf_inspect
def get_restore_function(registered_name):
    """Returns restore function registered to name."""
    return _saver_registry.name_lookup(registered_name)[1]