from __future__ import absolute_import
import re
import glob
import os
import os.path
def numsplit(text):
    """    Convert string into a list of texts and numbers in order to support a
    natural sorting.
    """
    result = []
    for group in re.split('(\\d+)', text):
        if group:
            try:
                group = int(group)
            except ValueError:
                pass
            result.append(group)
    return result