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
def fix_symlink_if_necessary(src_dir, dst_dir):
    logger.info('scanning for internal symlinks that point to the original virtual env')
    for dirpath, dirnames, filenames in os.walk(dst_dir):
        for a_file in itertools.chain(filenames, dirnames):
            full_file_path = os.path.join(dirpath, a_file)
            if os.path.islink(full_file_path):
                target = os.path.realpath(full_file_path)
                if target.startswith(src_dir):
                    new_target = target.replace(src_dir, dst_dir)
                    logger.debug('fixing symlink in %s' % (full_file_path,))
                    os.remove(full_file_path)
                    os.symlink(new_target, full_file_path)