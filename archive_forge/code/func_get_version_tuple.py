import datetime
import functools
import os
import subprocess
import sys
from django.utils.regex_helper import _lazy_re_compile
def get_version_tuple(version):
    """
    Return a tuple of version numbers (e.g. (1, 2, 3)) from the version
    string (e.g. '1.2.3').
    """
    version_numbers = []
    for item in version_component_re.split(version):
        if item and item != '.':
            try:
                component = int(item)
            except ValueError:
                break
            else:
                version_numbers.append(component)
    return tuple(version_numbers)