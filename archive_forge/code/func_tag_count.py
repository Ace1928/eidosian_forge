import pickle
import re
from debian.deprecation import function_deprecated_by
def tag_count(self):
    """Return the number of tags"""
    return len(self.rdb)