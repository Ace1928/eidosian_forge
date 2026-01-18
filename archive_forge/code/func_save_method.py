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
def save_method(elements, module_path):
    """ Recursively save methods with module name and signature. """
    for elem, signature in elements.items():
        if isinstance(signature, dict):
            save_method(signature, module_path + (elem,))
        elif isinstance(signature, Class):
            save_method(signature.fields, module_path + (elem,))
        elif signature.ismethod():
            if elem in MODULES['__dispatch__'] and module_path[0] != '__dispatch__':
                duplicated_methods.setdefault(elem, []).append((module_path, signature))
            if elem in methods and module_path[0] != '__dispatch__':
                assert elem in MODULES['__dispatch__']
                path = ('__dispatch__',)
                methods[elem] = (path, MODULES['__dispatch__'][elem])
            else:
                methods[elem] = (module_path, signature)