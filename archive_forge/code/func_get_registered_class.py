import collections
import re
from tensorflow.python.util import tf_inspect
def get_registered_class(registered_name):
    try:
        return _class_registry.name_lookup(registered_name)
    except LookupError:
        return None