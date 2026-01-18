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
def fixup_pth_file(filename, old_dir, new_dir):
    logger.debug('fixup_pth_file %s' % filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    has_change = False
    for num, line in enumerate(lines):
        line = (line.decode('utf-8') if hasattr(line, 'decode') else line).strip()
        if not line or line.startswith('#') or line.startswith('import '):
            continue
        elif _dirmatch(line, old_dir):
            lines[num] = line.replace(old_dir, new_dir, 1)
            has_change = True
    if has_change:
        with open(filename, 'w') as f:
            payload = os.linesep.join([line.strip() for line in lines]) + os.linesep
            f.write(payload)