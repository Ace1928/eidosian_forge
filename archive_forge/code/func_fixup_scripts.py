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
def fixup_scripts(old_dir, new_dir, version, rewrite_env_python=False):
    bin_dir = os.path.join(new_dir, env_bin_dir)
    root, dirs, files = next(os.walk(bin_dir))
    pybinre = re.compile('pythonw?([0-9]+(\\.[0-9]+(\\.[0-9]+)?)?)?$')
    for file_ in files:
        filename = os.path.join(root, file_)
        if file_ in ['python', 'python%s' % version, 'activate_this.py']:
            continue
        elif file_.startswith('python') and pybinre.match(file_):
            continue
        elif file_.endswith('.pyc'):
            continue
        elif file_ == 'activate' or file_.startswith('activate.'):
            fixup_activate(os.path.join(root, file_), old_dir, new_dir)
        elif os.path.islink(filename):
            fixup_link(filename, old_dir, new_dir)
        elif os.path.isfile(filename):
            fixup_script_(root, file_, old_dir, new_dir, version, rewrite_env_python=rewrite_env_python)