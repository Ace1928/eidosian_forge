import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def parseCmdLine(argv=(None,)):
    import optparse
    parser = optparse.OptionParser('usage: %prog [-v] [info objs]')
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='be verbose and print more messages')
    opts, args = parser.parse_args(args=argv[1:])
    return (opts, args)