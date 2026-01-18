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
def func_dependency(self, node: BaseNode, args: T.List[TYPE_var], kwargs: T.Dict[str, TYPE_var]) -> None:
    args = self.flatten_args(args)
    kwargs = self.flatten_kwargs(kwargs)
    if not args:
        return
    name = args[0]
    has_fallback = 'fallback' in kwargs
    required = kwargs.get('required', True)
    version = kwargs.get('version', [])
    if not isinstance(version, list):
        version = [version]
    if isinstance(required, ElementaryNode):
        required = required.value
    if not isinstance(required, bool):
        required = False
    self.dependencies += [{'name': name, 'required': required, 'version': version, 'has_fallback': has_fallback, 'conditional': node.condition_level > 0, 'node': node}]