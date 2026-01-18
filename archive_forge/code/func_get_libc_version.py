import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_libc_version():
    import platform
    if get_platform() != 'linux':
        return 'N/A'
    return '-'.join(platform.libc_ver())