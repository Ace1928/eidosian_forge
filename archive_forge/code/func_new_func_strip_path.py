import io
import os
import os.path
import sys
import warnings
import cherrypy
def new_func_strip_path(func_name):
    """Add ``__init__`` modules' parents.

        This makes the profiler output more readable.
        """
    filename, line, name = func_name
    if filename.endswith('__init__.py'):
        return (os.path.basename(filename[:-12]) + filename[-12:], line, name)
    return (os.path.basename(filename), line, name)