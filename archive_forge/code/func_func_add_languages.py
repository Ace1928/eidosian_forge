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
def func_add_languages(self, node: BaseNode, args: T.List[TYPE_var], kwargs: T.Dict[str, TYPE_var]) -> None:
    kwargs = self.flatten_kwargs(kwargs)
    required = kwargs.get('required', True)
    assert isinstance(required, (bool, cdata.UserFeatureOption)), 'for mypy'
    if isinstance(required, cdata.UserFeatureOption):
        required = required.is_enabled()
    if 'native' in kwargs:
        native = kwargs.get('native', False)
        self._add_languages(args, required, MachineChoice.BUILD if native else MachineChoice.HOST)
    else:
        for for_machine in [MachineChoice.BUILD, MachineChoice.HOST]:
            self._add_languages(args, required, for_machine)