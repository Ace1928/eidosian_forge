import os
import sys
import time
import errno
import shutil
from multiprocessing import util
def memstr_to_bytes(text):
    """ Convert a memory text to its value in bytes.
    """
    kilo = 1024
    units = dict(K=kilo, M=kilo ** 2, G=kilo ** 3)
    try:
        size = int(units[text[-1]] * float(text[:-1]))
    except (KeyError, ValueError) as e:
        raise ValueError("Invalid literal for size give: %s (type %s) should be alike '10G', '500M', '50K'." % (text, type(text))) from e
    return size