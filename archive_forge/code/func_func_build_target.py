from __future__ import annotations
import copy
import os
import typing as T
from .. import compilers, environment, mesonlib, optinterpreter
from .. import coredata as cdata
from ..build import Executable, Jar, SharedLibrary, SharedModule, StaticLibrary
from ..compilers import detect_compiler_for
from ..interpreterbase import InvalidArguments, SubProject
from ..mesonlib import MachineChoice, OptionKey
from ..mparser import BaseNode, ArithmeticNode, ArrayNode, ElementaryNode, IdNode, FunctionNode, BaseStringNode
from .interpreter import AstInterpreter
def func_build_target(self, node: BaseNode, args: T.List[TYPE_var], kwargs: T.Dict[str, TYPE_var]) -> T.Optional[T.Dict[str, T.Any]]:
    if 'target_type' not in kwargs:
        return None
    target_type = kwargs.pop('target_type')
    if isinstance(target_type, ElementaryNode):
        target_type = target_type.value
    if target_type == 'executable':
        return self.build_target(node, args, kwargs, Executable)
    elif target_type == 'shared_library':
        return self.build_target(node, args, kwargs, SharedLibrary)
    elif target_type == 'static_library':
        return self.build_target(node, args, kwargs, StaticLibrary)
    elif target_type == 'both_libraries':
        return self.build_target(node, args, kwargs, SharedLibrary)
    elif target_type == 'library':
        return self.build_library(node, args, kwargs)
    elif target_type == 'jar':
        return self.build_target(node, args, kwargs, Jar)
    return None