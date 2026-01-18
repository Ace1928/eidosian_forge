import os
import warnings
from .errors import (
from .ccompiler import CCompiler, gen_preprocess_options
from .file_util import write_file
from ._modified import newer
from ._log import log
def create_static_lib(self, objects, output_libname, output_dir=None, debug=0, target_lang=None):
    objects, output_dir = self._fix_object_args(objects, output_dir)
    output_filename = self.library_filename(output_libname, output_dir=output_dir)
    if self._need_link(objects, output_filename):
        lib_args = [output_filename, '/u'] + objects
        if debug:
            pass
        try:
            self.spawn([self.lib] + lib_args)
        except DistutilsExecError as msg:
            raise LibError(msg)
    else:
        log.debug('skipping %s (up-to-date)', output_filename)