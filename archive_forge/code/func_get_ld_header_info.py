import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def get_ld_header_info(p):
    info = []
    for line in p.stdout:
        if re.match('[0-9]', line):
            info.append(line)
        else:
            break
    return info