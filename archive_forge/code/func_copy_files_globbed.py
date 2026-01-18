from gitdb import OStream
import sys
import random
from array import array
from io import BytesIO
import glob
import unittest
import tempfile
import shutil
import os
import gc
import logging
from functools import wraps
def copy_files_globbed(source_glob, target_dir, hard_link_ok=False):
    """Copy all files found according to the given source glob into the target directory
    :param hard_link_ok: if True, hard links will be created if possible. Otherwise
        the files will be copied"""
    for src_file in glob.glob(source_glob):
        if hard_link_ok and hasattr(os, 'link'):
            target = os.path.join(target_dir, os.path.basename(src_file))
            try:
                os.link(src_file, target)
            except OSError:
                shutil.copy(src_file, target_dir)
        else:
            shutil.copy(src_file, target_dir)