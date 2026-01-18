import pickle
import re
from debian.deprecation import function_deprecated_by
def iter_tags_packages(self):
    """Iterate over 2-tuples of (tag, pkgs)"""
    return self.rdb.items()