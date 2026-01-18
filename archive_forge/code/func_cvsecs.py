import re
import threading
import time
from ._utils import logger
def cvsecs(*args):
    """converts a time to second. Either cvsecs(min, secs) or
    cvsecs(hours, mins, secs).
    """
    if len(args) == 1:
        return float(args[0])
    elif len(args) == 2:
        return 60 * float(args[0]) + float(args[1])
    elif len(args) == 3:
        return 3600 * float(args[0]) + 60 * float(args[1]) + float(args[2])