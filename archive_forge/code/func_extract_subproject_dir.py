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
def extract_subproject_dir(self) -> T.Optional[str]:
    """Fast path to extract subproject_dir kwarg.
           This is faster than self.parse_project() which also initialize options
           and also calls parse_project() on every subproject.
        """
    if not self.ast.lines:
        return None
    project = self.ast.lines[0]
    if not isinstance(project, FunctionNode):
        return None
    for kw, val in project.args.kwargs.items():
        assert isinstance(kw, IdNode), 'for mypy'
        if kw.value == 'subproject_dir':
            if isinstance(val, BaseStringNode):
                return val.value
    return None