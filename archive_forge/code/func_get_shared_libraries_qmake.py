import os, glob, re, sys
from distutils import sysconfig
def get_shared_libraries_qmake(which_package):
    libs = get_shared_libraries_data(which_package)
    if libs is None:
        return None
    if sys.platform == 'win32':
        if not libs:
            return ''
        dlls = ''
        for lib in libs:
            dll = os.path.splitext(lib)[0] + '.dll'
            dlls += dll + ' '
        return dlls
    else:
        libs_string = ''
        for lib in libs:
            libs_string += lib + ' '
        return libs_string