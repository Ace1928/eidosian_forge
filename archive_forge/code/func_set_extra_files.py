from distutils import errors
import os
def set_extra_files(extra_files):
    for filename in extra_files:
        if not os.path.exists(filename):
            raise errors.DistutilsFileError('%s from the extra_files option in setup.cfg does not exist' % filename)
    global _extra_files
    _extra_files[:] = extra_files[:]