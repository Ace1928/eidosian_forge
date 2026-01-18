import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def reset_global_f2py_vars():
    global groupcounter, grouplist, neededmodule, expectbegin
    global skipblocksuntil, usermodules, f90modulevars, gotnextfile
    global filepositiontext, currentfilename, skipfunctions, skipfuncs
    global onlyfuncs, include_paths, previous_context
    global strictf77, sourcecodeform, quiet, verbose, tabchar, pyffilename
    global f77modulename, skipemptyends, ignorecontains, dolowercase, debug
    strictf77 = 1
    sourcecodeform = 'fix'
    quiet = 0
    verbose = 1
    tabchar = 4 * ' '
    pyffilename = ''
    f77modulename = ''
    skipemptyends = 0
    ignorecontains = 1
    dolowercase = 1
    debug = []
    groupcounter = 0
    grouplist = {groupcounter: []}
    neededmodule = -1
    expectbegin = 1
    skipblocksuntil = -1
    usermodules = []
    f90modulevars = {}
    gotnextfile = 1
    filepositiontext = ''
    currentfilename = ''
    skipfunctions = []
    skipfuncs = []
    onlyfuncs = []
    include_paths = []
    previous_context = None