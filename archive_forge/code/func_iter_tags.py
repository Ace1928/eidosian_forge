import pickle
import re
from debian.deprecation import function_deprecated_by
def iter_tags(self):
    """Iterate over the tags"""
    return self.rdb.keys()