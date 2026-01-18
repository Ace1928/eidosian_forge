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
def fixup_syspath_items(syspath, old_dir, new_dir):
    for path in syspath:
        if not os.path.isdir(path):
            continue
        path = os.path.normcase(os.path.abspath(path))
        if _dirmatch(path, old_dir):
            path = path.replace(old_dir, new_dir, 1)
            if not os.path.exists(path):
                continue
        elif not _dirmatch(path, new_dir):
            continue
        root, dirs, files = next(os.walk(path))
        for file_ in files:
            filename = os.path.join(root, file_)
            if filename.endswith('.pth'):
                fixup_pth_file(filename, old_dir, new_dir)
            elif filename.endswith('.egg-link'):
                fixup_egglink_file(filename, old_dir, new_dir)