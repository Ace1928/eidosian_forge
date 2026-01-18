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
def do_subproject(self, dirname: SubProject) -> None:
    subproject_dir_abs = os.path.join(self.environment.get_source_dir(), self.subproject_dir)
    subpr = os.path.join(subproject_dir_abs, dirname)
    try:
        subi = IntrospectionInterpreter(subpr, '', self.backend, cross_file=self.cross_file, subproject=dirname, subproject_dir=self.subproject_dir, env=self.environment, visitors=self.visitors)
        subi.analyze()
        subi.project_data['name'] = dirname
        self.project_data['subprojects'] += [subi.project_data]
    except (mesonlib.MesonException, RuntimeError):
        return