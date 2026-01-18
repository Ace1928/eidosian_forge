from __future__ import with_statement
import logging
import optparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import itertools
def fixup_link(filename, old_dir, new_dir, target=None):
    logger.debug('fixing %s' % filename)
    if target is None:
        target = os.readlink(filename)
    origdir = os.path.dirname(os.path.abspath(filename)).replace(new_dir, old_dir)
    if not os.path.isabs(target):
        target = os.path.abspath(os.path.join(origdir, target))
        rellink = True
    else:
        rellink = False
    if _dirmatch(target, old_dir):
        if rellink:
            target = target[len(origdir):].lstrip(os.sep)
        else:
            target = target.replace(old_dir, new_dir, 1)
    _replace_symlink(filename, target)