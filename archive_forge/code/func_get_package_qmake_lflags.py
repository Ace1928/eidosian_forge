import os, glob, re, sys
from distutils import sysconfig
def get_package_qmake_lflags(which_package):
    package_path = find_package(which_package)
    if package_path is None:
        return None
    link = '-L{}'.format(package_path)
    glob_result = glob.glob(os.path.join(package_path, shared_library_glob_pattern()))
    for lib in filter_shared_libraries(glob_result):
        link += ' '
        link += link_option(lib)
    return link