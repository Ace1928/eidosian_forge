import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def change_user_group(self, user, group):
    if not user and (not group):
        return
    import pwd, grp
    uid = gid = None
    if group:
        try:
            gid = int(group)
            group = grp.getgrgid(gid).gr_name
        except ValueError:
            import grp
            try:
                entry = grp.getgrnam(group)
            except KeyError:
                raise BadCommand('Bad group: %r; no such group exists' % group)
            gid = entry.gr_gid
    try:
        uid = int(user)
        user = pwd.getpwuid(uid).pw_name
    except ValueError:
        try:
            entry = pwd.getpwnam(user)
        except KeyError:
            raise BadCommand('Bad username: %r; no such user exists' % user)
        if not gid:
            gid = entry.pw_gid
        uid = entry.pw_uid
    if self.verbose > 0:
        print('Changing user to %s:%s (%s:%s)' % (user, group or '(unknown)', uid, gid))
    if hasattr(os, 'initgroups'):
        os.initgroups(user, gid)
    else:
        os.setgroups([e.gr_gid for e in grp.getgrall() if user in e.gr_mem] + [gid])
    if gid:
        os.setgid(gid)
    if uid:
        os.setuid(uid)