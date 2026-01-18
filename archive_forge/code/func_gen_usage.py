import os
import sys
from distutils.debug import DEBUG
from distutils.errors import *
from distutils.dist import Distribution
from distutils.cmd import Command
from distutils.config import PyPIRCCommand
from distutils.extension import Extension
def gen_usage(script_name):
    script = os.path.basename(script_name)
    return USAGE % vars()