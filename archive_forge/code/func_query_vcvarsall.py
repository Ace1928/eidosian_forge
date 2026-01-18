import os
import subprocess
import sys
import re
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
import winreg
def query_vcvarsall(version, arch='x86'):
    """Launch vcvarsall.bat and read the settings from its environment
    """
    vcvarsall = find_vcvarsall(version)
    interesting = {'include', 'lib', 'libpath', 'path'}
    result = {}
    if vcvarsall is None:
        raise DistutilsPlatformError('Unable to find vcvarsall.bat')
    log.debug("Calling 'vcvarsall.bat %s' (version=%s)", arch, version)
    popen = subprocess.Popen('"%s" %s & set' % (vcvarsall, arch), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = popen.communicate()
        if popen.wait() != 0:
            raise DistutilsPlatformError(stderr.decode('mbcs'))
        stdout = stdout.decode('mbcs')
        for line in stdout.split('\n'):
            line = Reg.convert_mbcs(line)
            if '=' not in line:
                continue
            line = line.strip()
            key, value = line.split('=', 1)
            key = key.lower()
            if key in interesting:
                if value.endswith(os.pathsep):
                    value = value[:-1]
                result[key] = removeDuplicates(value)
    finally:
        popen.stdout.close()
        popen.stderr.close()
    if len(result) != len(interesting):
        raise ValueError(str(list(result.keys())))
    return result