import os
import re
import sys
import platform
from .compat import string_types
from .util import in_venv, parse_marker
from .version import LegacyVersion as LV
def default_context():

    def format_full_version(info):
        version = '%s.%s.%s' % (info.major, info.minor, info.micro)
        kind = info.releaselevel
        if kind != 'final':
            version += kind[0] + str(info.serial)
        return version
    if hasattr(sys, 'implementation'):
        implementation_version = format_full_version(sys.implementation.version)
        implementation_name = sys.implementation.name
    else:
        implementation_version = '0'
        implementation_name = ''
    ppv = platform.python_version()
    m = _DIGITS.match(ppv)
    pv = m.group(0)
    result = {'implementation_name': implementation_name, 'implementation_version': implementation_version, 'os_name': os.name, 'platform_machine': platform.machine(), 'platform_python_implementation': platform.python_implementation(), 'platform_release': platform.release(), 'platform_system': platform.system(), 'platform_version': platform.version(), 'platform_in_venv': str(in_venv()), 'python_full_version': ppv, 'python_version': pv, 'sys_platform': sys.platform}
    return result