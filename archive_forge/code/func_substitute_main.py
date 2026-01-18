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
def substitute_main(name, cmds=None, args=None):
    """
    If module has no other commands, use this function to add all of the ones in pyct.cmd
    """
    parser = argparse.ArgumentParser(description='%s commands' % name)
    subparsers = parser.add_subparsers(title='available commands')
    add_commands(subparsers, name, cmds, args)
    add_version(parser, name)
    args = parser.parse_args()
    args.func(args) if hasattr(args, 'func') else parser.error('must supply command to run')