import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def get_script_home():
    """Get the directory containing the program entry module.

    For ordinary Python scripts, this is the directory containing the
    ``__main__`` module.  For executables created with py2exe the result is
    the directory containing the running executable file.  For OS X bundles
    created using Py2App the result is the Resources directory within the
    running bundle.

    If none of the above cases apply and the file for ``__main__`` cannot
    be determined the working directory is returned.

    When the script is being run by a Python profiler, this function
    may return the directory where the profiler is running instead of
    the directory of the real script. To workaround this behaviour the
    full path to the real script can be specified in :py:attr:`pyglet.resource.path`.

    :rtype: str
    """
    frozen = getattr(sys, 'frozen', None)
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        return meipass
    elif frozen in ('windows_exe', 'console_exe'):
        return os.path.dirname(sys.executable)
    elif frozen == 'macosx_app':
        return os.environ['RESOURCEPATH']
    else:
        main = sys.modules['__main__']
        if hasattr(main, '__file__'):
            return os.path.dirname(os.path.abspath(main.__file__))
        elif 'python' in os.path.basename(sys.executable):
            return os.getcwd()
        else:
            return os.path.dirname(sys.executable)