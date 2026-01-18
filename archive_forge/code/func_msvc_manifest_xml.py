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
def msvc_manifest_xml(maj, min):
    """Given a major and minor version of the MSVCR, returns the
    corresponding XML file."""
    try:
        fullver = _MSVCRVER_TO_FULLVER[str(maj * 10 + min)]
    except KeyError:
        raise ValueError('Version %d,%d of MSVCRT not supported yet' % (maj, min)) from None
    template = textwrap.dedent('        <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n          <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n            <security>\n              <requestedPrivileges>\n                <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n              </requestedPrivileges>\n            </security>\n          </trustInfo>\n          <dependency>\n            <dependentAssembly>\n              <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>\n            </dependentAssembly>\n          </dependency>\n        </assembly>')
    return template % {'fullver': fullver, 'maj': maj, 'min': min}