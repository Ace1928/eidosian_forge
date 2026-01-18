import pickle
import re
from debian.deprecation import function_deprecated_by
def tags_of_package(self, pkg):
    """Return the tag set of a package"""
    return self.db[pkg] if pkg in self.db else set()