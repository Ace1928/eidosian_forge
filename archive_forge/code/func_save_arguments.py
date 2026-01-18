import gast as ast
from importlib import import_module
import inspect
import logging
import numpy
import sys
from pythran.typing import Dict, Set, List, TypeVar, Union, Optional, NDArray
from pythran.typing import Generator, Fun, Tuple, Iterable, Sized, File
from pythran.conversion import to_ast, ToNotEval
from pythran.intrinsic import Class
from pythran.intrinsic import ClassWithConstConstructor, ExceptionClass
from pythran.intrinsic import ClassWithReadOnceConstructor
from pythran.intrinsic import ConstFunctionIntr, FunctionIntr, UpdateEffect
from pythran.intrinsic import ConstMethodIntr, MethodIntr
from pythran.intrinsic import AttributeIntr, StaticAttributeIntr
from pythran.intrinsic import ReadEffect, ConstantIntr, UFunc
from pythran.intrinsic import ReadOnceMethodIntr
from pythran.intrinsic import ReadOnceFunctionIntr, ConstExceptionIntr
from pythran import interval
import beniget
def save_arguments(module_name, elements):
    """ Recursively save arguments name and default value. """
    for elem, signature in elements.items():
        if isinstance(signature, dict):
            save_arguments(module_name + (elem,), signature)
        else:
            try:
                themodule = import_module('.'.join(module_name))
                obj = getattr(themodule, elem)
                while hasattr(obj, '__wrapped__'):
                    obj = obj.__wrapped__
            except (AttributeError, ImportError, TypeError):
                continue
            try:
                spec = inspect.getfullargspec(obj)
            except:
                continue
            args = [ast.Name(arg, ast.Param(), None, None) for arg in spec.args]
            defaults = list(spec.defaults or [])
            args += [ast.Name(arg, ast.Param(), None, None) for arg in spec.kwonlyargs]
            defaults += [spec.kwonlydefaults[kw] for kw in spec.kwonlyargs]
            if signature.args.args:
                if module_name != ('numpy', 'random'):
                    logger.warning('Overriding pythran description with argspec information for: {}'.format('.'.join(module_name + (elem,))))
                else:
                    continue
            signature_args = args[:-len(defaults) or None]
            signature_defaults = []
            try:
                for arg, value in zip(args[-len(defaults):], defaults):
                    signature_args.append(arg)
                    signature_defaults.append(to_ast(value))
            except ToNotEval:
                continue
            signature.args.args = signature_args
            signature.args.defaults = signature_defaults