import os
import sys
import ctypes
def generate_candidate_libs(lib_names, lib_dirs=None):
    """Generate a list of candidate filenames of what might be the dynamic
    library corresponding with the given list of names.
    Returns (lib_dirs, lib_paths)
    """
    lib_dirs = lib_dirs or []
    sys_lib_dirs = ['/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu', '/usr/lib/aarch64-linux-gnu', '/usr/local/lib', '/opt/local/lib']
    py_sub_dirs = ['bin', 'lib', 'DLLs', 'Library/bin', 'shared']
    py_lib_dirs = [os.path.join(sys.prefix, d) for d in py_sub_dirs]
    if hasattr(sys, 'base_prefix'):
        py_lib_dirs += [os.path.join(sys.base_prefix, d) for d in py_sub_dirs]
    home_dir = os.path.expanduser('~')
    user_lib_dirs = [os.path.join(home_dir, d) for d in ['lib']]
    potential_lib_dirs = lib_dirs + sys_lib_dirs + py_lib_dirs + user_lib_dirs
    lib_dirs = []
    for ld in potential_lib_dirs:
        if os.path.isdir(ld) and ld not in lib_dirs:
            lib_dirs.append(ld)
    lib_paths = []
    for lib_dir in lib_dirs:
        files = os.listdir(lib_dir)
        files = reversed(sorted(files))
        files = sorted(files, key=len)
        for lib_name in lib_names:
            for fname in files:
                if fname.lower().startswith(lib_name) and looks_lib(fname):
                    lib_paths.append(os.path.join(lib_dir, fname))
    lib_paths = [lp for lp in lib_paths if os.path.isfile(lp)]
    return (lib_dirs, lib_paths)