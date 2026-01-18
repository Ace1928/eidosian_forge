import os, glob, re, sys
from distutils import sysconfig
def filter_shared_libraries(libs_list):

    def predicate(lib_name):
        basename = os.path.basename(lib_name)
        if 'shiboken' in basename or 'pyside2' in basename:
            return True
        return False
    result = [lib for lib in libs_list if predicate(lib)]
    return result