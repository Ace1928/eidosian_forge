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
def generate_manifest(config):
    msver = get_build_msvc_version()
    if msver is not None:
        if msver >= 8:
            check_embedded_msvcr_match_linked(msver)
            ma_str, mi_str = str(msver).split('.')
            manxml = msvc_manifest_xml(int(ma_str), int(mi_str))
            with open(manifest_name(config), 'w') as man:
                config.temp_files.append(manifest_name(config))
                man.write(manxml)