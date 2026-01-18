import collections
import re
from tensorflow.python.util import tf_inspect
def get_registered_class_name(obj):
    try:
        return _class_registry.get_registered_name(obj)
    except LookupError:
        return None