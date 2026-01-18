from pythran.backend import Cxx, Python
from pythran.config import cfg
from pythran.cxxgen import PythonModule, Include, Line, Statement
from pythran.cxxgen import FunctionBody, FunctionDeclaration, Value, Block
from pythran.cxxgen import ReturnStatement
from pythran.dist import PythranExtension, PythranBuildExt
from pythran.middlend import refine, mark_unexported_functions
from pythran.passmanager import PassManager
from pythran.tables import pythran_ward
from pythran.types import tog
from pythran.types.type_dependencies import pytype_to_deps
from pythran.types.conversion import pytype_to_ctype
from pythran.spec import load_specfile, Spec
from pythran.spec import spec_to_string
from pythran.syntax import check_specs, check_exports, PythranSyntaxError
from pythran.version import __version__
from pythran.utils import cxxid
import pythran.frontend as frontend
from tempfile import mkdtemp, NamedTemporaryFile
import gast as ast
import importlib
import logging
import os.path
import shutil
import glob
import hashlib
from functools import reduce
import sys
def compile_pythranfile(file_path, output_file=None, module_name=None, cpponly=False, pyonly=False, report_times=False, **kwargs):
    """
    Pythran file -> c++ file -> native module.

    Returns the generated .so (or .cpp if `cpponly` is set to true).

    Usage without an existing spec file

    >>> with open('pythran_test.py', 'w') as fd:
    ...    _ = fd.write('def foo(i): return i ** 2')
    >>> cpp_path = compile_pythranfile('pythran_test.py', cpponly=True)

    Usage with an existing spec file:

    >>> with open('pythran_test.pythran', 'w') as fd:
    ...    _ = fd.write('export foo(int)')
    >>> so_path = compile_pythranfile('pythran_test.py')

    Specify the output file:

    >>> import sysconfig
    >>> ext = sysconfig.get_config_vars()["EXT_SUFFIX"]
    >>> so_path = compile_pythranfile('pythran_test.py', output_file='foo'+ext)
    """
    if not output_file:
        _, basename = os.path.split(file_path)
        module_name = module_name or os.path.splitext(basename)[0]
    else:
        _, basename = os.path.split(output_file.replace('%{ext}', ''))
        module_name = module_name or basename.split('.', 1)[0]
    module_dir = os.path.dirname(file_path)
    spec_file = os.path.splitext(file_path)[0] + '.pythran'
    if os.path.isfile(spec_file):
        specs = load_specfile(spec_file)
        kwargs.setdefault('specs', specs)
    try:
        with open(file_path) as fd:
            output_file = compile_pythrancode(module_name, fd.read(), output_file=output_file, cpponly=cpponly, pyonly=pyonly, module_dir=module_dir, report_times=report_times, **kwargs)
    except PythranSyntaxError as e:
        if e.filename is None:
            e.filename = file_path
        raise
    return output_file