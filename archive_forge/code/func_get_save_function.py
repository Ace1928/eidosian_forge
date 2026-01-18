import collections
import re
from tensorflow.python.util import tf_inspect
def get_save_function(registered_name):
    """Returns save function registered to name."""
    return _saver_registry.name_lookup(registered_name)[0]