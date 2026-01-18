from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def appdata_dir(appname=None, roaming=False):
    """appdata_dir(appname=None, roaming=False)

    Get the path to the application directory, where applications are allowed
    to write user specific files (e.g. configurations). For non-user specific
    data, consider using common_appdata_dir().
    If appname is given, a subdir is appended (and created if necessary).
    If roaming is True, will prefer a roaming directory (Windows Vista/7).
    """
    userDir = os.getenv('IMAGEIO_USERDIR', None)
    if userDir is None:
        userDir = os.path.expanduser('~')
        if not os.path.isdir(userDir):
            userDir = '/var/tmp'
    path = None
    if sys.platform.startswith('win'):
        path1, path2 = (os.getenv('LOCALAPPDATA'), os.getenv('APPDATA'))
        path = path2 or path1 if roaming else path1 or path2
    elif sys.platform.startswith('darwin'):
        path = os.path.join(userDir, 'Library', 'Application Support')
    if not (path and os.path.isdir(path)):
        path = userDir
    prefix = sys.prefix
    if getattr(sys, 'frozen', None):
        prefix = os.path.abspath(os.path.dirname(sys.executable))
    for reldir in ('settings', '../settings'):
        localpath = os.path.abspath(os.path.join(prefix, reldir))
        if os.path.isdir(localpath):
            try:
                open(os.path.join(localpath, 'test.write'), 'wb').close()
                os.remove(os.path.join(localpath, 'test.write'))
            except IOError:
                pass
            else:
                path = localpath
                break
    if appname:
        if path == userDir:
            appname = '.' + appname.lstrip('.')
        path = os.path.join(path, appname)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
    return path