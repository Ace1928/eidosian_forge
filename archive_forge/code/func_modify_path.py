import os
import sys
import pkg_resources
def modify_path():
    """Modify the module search path."""
    if os.name != 'nt':
        return
    path = os.environ.get('PATH')
    if path is None:
        return
    try:
        extra_dll_dir = pkg_resources.resource_filename('google_crc32c', 'extra-dll')
        if os.path.isdir(extra_dll_dir):
            os.environ['PATH'] = path + os.pathsep + extra_dll_dir
            if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
                os.add_dll_directory(extra_dll_dir)
    except ImportError:
        pass