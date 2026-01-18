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
def compile_pythrancode(module_name, pythrancode, specs=None, opts=None, cpponly=False, pyonly=False, output_file=None, module_dir=None, report_times=False, **kwargs):
    """Pythran code (string) -> c++ code -> native module

    if `cpponly` is set to true, return the generated C++ filename
    if `pyonly` is set to true, prints the generated Python filename,
       unless `output_file` is set
    otherwise, return the generated native library filename
    """
    if pyonly:
        content = generate_py(module_name, pythrancode, opts, module_dir, report_times)
        if output_file is None:
            print(content)
            return None
        else:
            tmp_file = _write_temp(content, '.py')
            output_file = output_file.format('.py')
            shutil.move(tmp_file, output_file)
            logger.info('Generated Python source file: ' + output_file)
    from pythran.spec import spec_parser
    if specs is None:
        specs = spec_parser(pythrancode)
    module, error_checker = generate_cxx(module_name, pythrancode, specs, opts, module_dir, report_times)
    if 'ENABLE_PYTHON_MODULE' in kwargs.get('undef_macros', []):
        module.preamble.insert(0, Line('#undef ENABLE_PYTHON_MODULE'))
        module.preamble.insert(0, Line('#define PY_MAJOR_VERSION {}'.format(sys.version_info.major)))
    if cpponly:
        tmp_file = _write_temp(str(module), '.cpp')
        if output_file:
            output_file = output_file.replace('%{ext}', '.cpp')
        else:
            output_file = module_name + '.cpp'
        shutil.move(tmp_file, output_file)
        logger.info('Generated C++ source file: ' + output_file)
    else:
        if not specs:
            raise ValueError('Empty spec files while generating native module')
        try:
            output_file = compile_cxxcode(module_name, str(module), output_binary=output_file, **kwargs)
        except CompileError:
            logger.warning('Compilation error, trying hard to find its origin...')
            error_checker()
            logger.warning("Nope, I'm going to flood you with C++ errors!")
            raise
    return output_file