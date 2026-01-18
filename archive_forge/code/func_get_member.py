import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def get_member(name, members):
    """
    Return an archive member matching the request in name.
    Name is the library name without any prefix like lib, suffix like .so,
    or version number.
    Given a list of members find and return the most appropriate result
    Priority is given to generic libXXX.so, then a versioned libXXX.so.a.b.c
    and finally, legacy AIX naming scheme.
    """
    expr = f'lib{name}\\.so'
    member = get_one_match(expr, members)
    if member:
        return member
    elif AIX_ABI == 64:
        expr = f'lib{name}64\\.so'
        member = get_one_match(expr, members)
    if member:
        return member
    member = get_version(name, members)
    if member:
        return member
    else:
        return get_legacy(members)