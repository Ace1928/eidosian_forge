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
def rewrite_shebang(version=None):
    logger.debug('fixing %s' % filename)
    shebang = new_shebang
    if version:
        shebang = shebang + version
    shebang = (shebang + '\n').encode('utf-8')
    with open(filename, 'wb') as f:
        f.write(shebang)
        f.writelines(lines[1:])