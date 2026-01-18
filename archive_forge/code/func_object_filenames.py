import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
    if output_dir is None:
        output_dir = ''
    obj_names = []
    for src_name in source_filenames:
        base, ext = os.path.splitext(os.path.normcase(src_name))
        drv, base = os.path.splitdrive(base)
        if drv:
            base = base[1:]
        if ext not in self.src_extensions + ['.rc', '.res']:
            raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
        if strip_dir:
            base = os.path.basename(base)
        if ext == '.res' or ext == '.rc':
            obj_names.append(os.path.join(output_dir, base + ext + self.obj_extension))
        else:
            obj_names.append(os.path.join(output_dir, base + self.obj_extension))
    return obj_names