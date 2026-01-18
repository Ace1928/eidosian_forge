import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def set_list_item(list_object, index_string, value):
    item_index = int(index_string)
    if len(list_object) <= item_index:
        list_object.extend([None] * (1 + item_index - len(list_object)))
    list_object[item_index] = value