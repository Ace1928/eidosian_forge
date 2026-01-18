from __future__ import print_function, absolute_import, division
from . import __version__
from .report import report
import os
import importlib
import inspect
import argparse
import distutils.dir_util
import shutil
from collections import OrderedDict
import glob
import sys
import tarfile
import time
import zipfile
import yaml
def copy_examples(name, path, verbose=False, force=False):
    """Copy examples to the supplied path."""
    source = _find_examples(name)
    path = os.path.abspath(path)
    if os.path.exists(path) and (not force):
        raise ValueError('Path %s already exists; please move it away, choose a different path, or use force.' % path)
    if verbose:
        print('Copying examples from %s' % source)
    distutils.dir_util.copy_tree(source, path, verbose=verbose)
    print('Copied examples to %s' % path)