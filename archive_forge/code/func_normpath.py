import os
import sys
import stat
import genericpath
from genericpath import *
def normpath(path):
    """Normalize path, eliminating double slashes, etc."""
    path = os.fspath(path)
    if isinstance(path, bytes):
        sep = b'/'
        empty = b''
        dot = b'.'
        dotdot = b'..'
    else:
        sep = '/'
        empty = ''
        dot = '.'
        dotdot = '..'
    if path == empty:
        return dot
    initial_slashes = path.startswith(sep)
    if initial_slashes and path.startswith(sep * 2) and (not path.startswith(sep * 3)):
        initial_slashes = 2
    comps = path.split(sep)
    new_comps = []
    for comp in comps:
        if comp in (empty, dot):
            continue
        if comp != dotdot or (not initial_slashes and (not new_comps)) or (new_comps and new_comps[-1] == dotdot):
            new_comps.append(comp)
        elif new_comps:
            new_comps.pop()
    comps = new_comps
    path = sep.join(comps)
    if initial_slashes:
        path = sep * initial_slashes + path
    return path or dot