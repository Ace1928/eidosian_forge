from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
def checkpoint_key(object_path, local_name):
    """Returns the checkpoint key for a local attribute of an object."""
    key_suffix = escape_local_name(local_name)
    if local_name == SERIALIZE_TO_TENSORS_NAME:
        key_suffix = ''
    return f'{object_path}/{OBJECT_ATTRIBUTES_NAME}/{key_suffix}'