import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def path_to_fspath(path, addat=True):
    _check_path(path)
    sp = path.strpath
    if addat and path.rev != -1:
        sp = '%s@%s' % (sp, path.rev)
    elif addat:
        sp = '%s@HEAD' % (sp,)
    return sp