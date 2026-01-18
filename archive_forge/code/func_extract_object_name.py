from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
def extract_object_name(key):
    """Substrings the checkpoint key to the start of "/.ATTRIBUTES"."""
    search_key = '/' + OBJECT_ATTRIBUTES_NAME
    return key[:key.index(search_key)]