import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
def default_aliases() -> t.List[t.Tuple[str, str]]:
    """Return list of shell aliases to auto-define.
    """
    if os.name == 'posix':
        default_aliases = [('mkdir', 'mkdir'), ('rmdir', 'rmdir'), ('mv', 'mv'), ('rm', 'rm'), ('cp', 'cp'), ('cat', 'cat')]
        if sys.platform.startswith('linux'):
            ls_aliases = [('ls', 'ls -F --color'), ('ll', 'ls -F -o --color'), ('lf', 'ls -F -o --color %l | grep ^-'), ('lk', 'ls -F -o --color %l | grep ^l'), ('ldir', 'ls -F -o --color %l | grep /$'), ('lx', 'ls -F -o --color %l | grep ^-..x')]
        elif sys.platform.startswith('openbsd') or sys.platform.startswith('netbsd'):
            ls_aliases = [('ls', 'ls -F'), ('ll', 'ls -F -l'), ('lf', 'ls -F -l %l | grep ^-'), ('lk', 'ls -F -l %l | grep ^l'), ('ldir', 'ls -F -l %l | grep /$'), ('lx', 'ls -F -l %l | grep ^-..x')]
        else:
            ls_aliases = [('ls', 'ls -F -G'), ('ll', 'ls -F -l -G'), ('lf', 'ls -F -l -G %l | grep ^-'), ('lk', 'ls -F -l -G %l | grep ^l'), ('ldir', 'ls -F -G -l %l | grep /$'), ('lx', 'ls -F -l -G %l | grep ^-..x')]
        default_aliases = default_aliases + ls_aliases
    elif os.name in ['nt', 'dos']:
        default_aliases = [('ls', 'dir /on'), ('ddir', 'dir /ad /on'), ('ldir', 'dir /ad /on'), ('mkdir', 'mkdir'), ('rmdir', 'rmdir'), ('echo', 'echo'), ('ren', 'ren'), ('copy', 'copy')]
    else:
        default_aliases = []
    return default_aliases