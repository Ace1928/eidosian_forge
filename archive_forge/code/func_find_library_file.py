import os
import warnings
from .errors import (
from .ccompiler import CCompiler, gen_preprocess_options
from .file_util import write_file
from ._modified import newer
from ._log import log
def find_library_file(self, dirs, lib, debug=0):
    if debug:
        dlib = lib + '_d'
        try_names = (dlib + '_bcpp', lib + '_bcpp', dlib, lib)
    else:
        try_names = (lib + '_bcpp', lib)
    for dir in dirs:
        for name in try_names:
            libfile = os.path.join(dir, self.library_filename(name))
            if os.path.exists(libfile):
                return libfile
    else:
        return None