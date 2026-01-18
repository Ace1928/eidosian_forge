import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def intern_release(name, releases=None):
    if releases is None:
        releases = _release_list
    return releases.get(name)