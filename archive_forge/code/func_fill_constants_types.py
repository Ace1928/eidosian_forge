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
def fill_constants_types(module_name, elements):
    """ Recursively save arguments name and default value. """
    for elem, intrinsic in elements.items():
        if isinstance(intrinsic, dict):
            fill_constants_types(module_name + (elem,), intrinsic)
        elif isinstance(intrinsic, ConstantIntr):
            cst = getattr(import_module('.'.join(module_name)), elem)
            intrinsic.signature = type(cst)