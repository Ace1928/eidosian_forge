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
def fixup_activate(filename, old_dir, new_dir):
    logger.debug('fixing %s' % filename)
    with open(filename, 'rb') as f:
        data = f.read().decode('utf-8')
    data = data.replace(old_dir, new_dir)
    with open(filename, 'wb') as f:
        f.write(data.encode('utf-8'))