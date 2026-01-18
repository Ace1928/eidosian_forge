import pickle
import re
from debian.deprecation import function_deprecated_by
def iter_packages(self):
    """Iterate over the packages"""
    return self.db.keys()