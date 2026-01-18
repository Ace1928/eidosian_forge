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
def check_embedded_msvcr_match_linked(msver):
    """msver is the ms runtime version used for the MANIFEST."""
    maj = msvc_runtime_major()
    if maj:
        if not maj == int(msver):
            raise ValueError('Discrepancy between linked msvcr (%d) and the one about to be embedded (%d)' % (int(msver), maj))