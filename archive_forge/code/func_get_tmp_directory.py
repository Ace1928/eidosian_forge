import os
import sys
import tempfile
from IPython.core.compilerop import CachingCompiler
def get_tmp_directory():
    """Get a temp directory."""
    tmp_dir = convert_to_long_pathname(tempfile.gettempdir())
    pid = os.getpid()
    return tmp_dir + os.sep + 'ipykernel_' + str(pid)