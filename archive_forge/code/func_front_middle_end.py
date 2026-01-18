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
def front_middle_end(module_name, code, optimizations=None, module_dir=None, entry_points=None, report_times=False):
    """Front-end and middle-end compilation steps"""
    pm = PassManager(module_name, module_dir, code)
    ir, docstrings = frontend.parse(pm, code)
    if entry_points is not None:
        ir = mark_unexported_functions(ir, entry_points)
    if optimizations is None:
        optimizations = cfg.get('pythran', 'optimizations').split()
    optimizations = [_parse_optimization(opt) for opt in optimizations]
    refine(pm, ir, optimizations, report_times)
    return (pm, ir, docstrings)