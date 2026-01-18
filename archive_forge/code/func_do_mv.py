import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def do_mv(self, test_case, input, args):
    err = None

    def error(msg, src, dst):
        return 'mv: cannot move {} to {}: {}\n'.format(src, dst, msg)
    if not args or len(args) != 2:
        raise SyntaxError('Usage: mv path1 path2')
    src, dst = args
    try:
        real_dst = dst
        if os.path.isdir(dst):
            real_dst = os.path.join(dst, os.path.basename(src))
        os.rename(src, real_dst)
    except OSError as e:
        if e.errno == errno.ENOENT:
            err = error('No such file or directory', src, dst)
        else:
            raise
    if err:
        retcode = 1
    else:
        retcode = 0
    return (retcode, None, err)