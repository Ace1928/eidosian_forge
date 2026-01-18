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
def generate_cxx(module_name, code, specs=None, optimizations=None, module_dir=None, report_times=False):
    """python + pythran spec -> c++ code
    returns a PythonModule object and an error checker

    the error checker can be used to print more detailed info on the origin of
    a compile error (e.g. due to bad typing)

    """
    if specs:
        entry_points = set(specs.keys())
    else:
        entry_points = None
    pm, ir, docstrings = front_middle_end(module_name, code, optimizations, module_dir, report_times=report_times, entry_points=entry_points)
    content = pm.dump(Cxx, ir)
    if specs is None:

        class Generable(object):

            def __init__(self, content):
                self.content = content

            def __str__(self):
                return str(self.content)
            generate = __str__
        mod = Generable(content)

        def error_checker():
            tog.typecheck(ir)
    else:
        if isinstance(specs, dict):
            specs = Spec(specs, {})

        def error_checker():
            types = tog.typecheck(ir)
            check_specs(specs, types)
        specs.to_docstrings(docstrings)
        check_exports(pm, ir, specs)
        if isinstance(code, bytes):
            code_bytes = code
        else:
            code_bytes = code.encode('ascii', 'ignore')
        metainfo = {'hash': hashlib.sha256(code_bytes).hexdigest(), 'version': __version__}
        mod = PythonModule(module_name, docstrings, metainfo)
        mod.add_to_includes(Include('pythonic/core.hpp'), Include('pythonic/python/core.hpp'), Include('pythonic/types/bool.hpp'), Include('pythonic/types/int.hpp'), Line('#ifdef _OPENMP\n#include <omp.h>\n#endif'))
        mod.add_to_includes(*[Include(inc) for inc in _extract_specs_dependencies(specs)])
        mod.add_to_includes(*content.body)
        mod.add_to_includes(Include('pythonic/python/exception_handler.hpp'))

        def warded(module_name, internal_name):
            return pythran_ward + '{0}::{1}'.format(module_name, internal_name)
        for function_name, signatures in specs.functions.items():
            internal_func_name = cxxid(function_name)
            if not signatures:
                mod.add_global_var(function_name, '{}()()'.format(warded(module_name, internal_func_name)))
            for sigid, signature in enumerate(signatures):
                numbered_function_name = '{0}{1}'.format(internal_func_name, sigid)
                arguments_types = [pytype_to_ctype(t) for t in signature]
                arguments_names = has_argument(ir, function_name)
                arguments = [n for n, _ in zip(arguments_names, arguments_types)]
                name_fmt = pythran_ward + '{0}::{1}::type{2}'
                args_list = ', '.join(arguments_types)
                specialized_fname = name_fmt.format(module_name, internal_func_name, '<{0}>'.format(args_list) if arguments_names else '')
                result_type = 'typename %s::result_type' % specialized_fname
                mod.add_pyfunction(FunctionBody(FunctionDeclaration(Value(result_type, numbered_function_name), [Value(t + '&&', a) for t, a in zip(arguments_types, arguments)]), Block([Statement('\n                            PyThreadState *_save = PyEval_SaveThread();\n                            try {{\n                                auto res = {0}()({1});\n                                PyEval_RestoreThread(_save);\n                                return res;\n                            }}\n                            catch(...) {{\n                                PyEval_RestoreThread(_save);\n                                throw;\n                            }}\n                            '.format(warded(module_name, internal_func_name), ', '.join(arguments)))])), function_name, arguments_types, signature)
        for function_name, signature in specs.capsules.items():
            internal_func_name = cxxid(function_name)
            arguments_types = [pytype_to_ctype(t) for t in signature]
            arguments_names = has_argument(ir, function_name)
            arguments = [n for n, _ in zip(arguments_names, arguments_types)]
            name_fmt = pythran_ward + '{0}::{1}::type{2}'
            args_list = ', '.join(arguments_types)
            specialized_fname = name_fmt.format(module_name, internal_func_name, '<{0}>'.format(args_list) if arguments_names else '')
            result_type = 'typename %s::result_type' % specialized_fname
            docstring = spec_to_string(function_name, signature)
            mod.add_capsule(FunctionBody(FunctionDeclaration(Value(result_type, function_name), [Value(t, a) for t, a in zip(arguments_types, arguments)]), Block([ReturnStatement('{0}()({1})'.format(warded(module_name, internal_func_name), ', '.join(arguments)))])), function_name, docstring)
    return (mod, error_checker)