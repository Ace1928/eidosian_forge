import os, glob, re, sys
from distutils import sysconfig
def get_shared_libraries_data(which_package):
    package_path = find_package(which_package)
    if package_path is None:
        return None
    glob_result = glob.glob(os.path.join(package_path, shared_library_glob_pattern()))
    filtered_libs = filter_shared_libraries(glob_result)
    libs = []
    if sys.platform == 'win32':
        for lib in filtered_libs:
            libs.append(os.path.realpath(lib))
    else:
        for lib in filtered_libs:
            libs.append(lib)
    return libs