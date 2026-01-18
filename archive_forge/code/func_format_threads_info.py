import os
import sys
import linecache
import re
import inspect
def format_threads_info():
    """ Returns a formatted string of the threads info.
    This can be useful in determining what's going on with created threads,
    especially when used in conjunction with greenlet
    """
    import threading
    threads = threading._active
    result = ['THREADS:']
    result.append(repr(threads))
    return os.linesep.join(result)