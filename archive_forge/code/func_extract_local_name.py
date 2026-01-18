from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
def extract_local_name(key, prefix=None):
    """Returns the substring after the "/.ATTIBUTES/" in the checkpoint key."""
    prefix = prefix or ''
    search_key = OBJECT_ATTRIBUTES_NAME + '/' + prefix
    try:
        return key[key.index(search_key) + len(search_key):]
    except ValueError:
        return key