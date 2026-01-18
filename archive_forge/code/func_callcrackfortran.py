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
def callcrackfortran(files, options):
    rules.options = options
    crackfortran.debug = options['debug']
    crackfortran.verbose = options['verbose']
    if 'module' in options:
        crackfortran.f77modulename = options['module']
    if 'skipfuncs' in options:
        crackfortran.skipfuncs = options['skipfuncs']
    if 'onlyfuncs' in options:
        crackfortran.onlyfuncs = options['onlyfuncs']
    crackfortran.include_paths[:] = options['include_paths']
    crackfortran.dolowercase = options['do-lower']
    postlist = crackfortran.crackfortran(files)
    if 'signsfile' in options:
        outmess('Saving signatures to file "%s"\n' % options['signsfile'])
        pyf = crackfortran.crack2fortran(postlist)
        if options['signsfile'][-6:] == 'stdout':
            sys.stdout.write(pyf)
        else:
            with open(options['signsfile'], 'w') as f:
                f.write(pyf)
    if options['coutput'] is None:
        for mod in postlist:
            mod['coutput'] = '%smodule.c' % mod['name']
    else:
        for mod in postlist:
            mod['coutput'] = options['coutput']
    if options['f2py_wrapper_output'] is None:
        for mod in postlist:
            mod['f2py_wrapper_output'] = '%s-f2pywrappers.f' % mod['name']
    else:
        for mod in postlist:
            mod['f2py_wrapper_output'] = options['f2py_wrapper_output']
    return postlist