from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def switchUID(uid, gid, euid=False):
    """
    Attempts to switch the uid/euid and gid/egid for the current process.

    If C{uid} is the same value as L{os.getuid} (or L{os.geteuid}),
    this function will issue a L{UserWarning} and not raise an exception.

    @type uid: C{int} or L{None}
    @param uid: the UID (or EUID) to switch the current process to. This
                parameter will be ignored if the value is L{None}.

    @type gid: C{int} or L{None}
    @param gid: the GID (or EGID) to switch the current process to. This
                parameter will be ignored if the value is L{None}.

    @type euid: C{bool}
    @param euid: if True, set only effective user-id rather than real user-id.
                 (This option has no effect unless the process is running
                 as root, in which case it means not to shed all
                 privileges, retaining the option to regain privileges
                 in cases such as spawning processes. Use with caution.)
    """
    if euid:
        setuid = os.seteuid
        setgid = os.setegid
        getuid = os.geteuid
    else:
        setuid = os.setuid
        setgid = os.setgid
        getuid = os.getuid
    if gid is not None:
        setgid(gid)
    if uid is not None:
        if uid == getuid():
            uidText = euid and 'euid' or 'uid'
            actionText = f'tried to drop privileges and set{uidText} {uid}'
            problemText = f'{uidText} is already {getuid()}'
            warnings.warn('{} but {}; should we be root? Continuing.'.format(actionText, problemText))
        else:
            initgroups(uid, gid)
            setuid(uid)