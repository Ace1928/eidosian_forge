import sys
import os
import pprint
import re
from pathlib import Path
from itertools import dropwhile
import argparse
import copy
from . import crackfortran
from . import rules
from . import cb_rules
from . import auxfuncs
from . import cfuncs
from . import f90mod_rules
from . import __version__
from . import capi_maps
from numpy.f2py._backends import f2py_build_generator
def make_f2py_compile_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dep', action='append', dest='dependencies')
    parser.add_argument('--backend', choices=['meson', 'distutils'], default='distutils')
    parser.add_argument('-m', dest='module_name')
    return parser