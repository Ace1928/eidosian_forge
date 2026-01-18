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
def func_project(self, node: BaseNode, args: T.List[TYPE_var], kwargs: T.Dict[str, TYPE_var]) -> None:
    if self.project_node:
        raise InvalidArguments('Second call to project()')
    self.project_node = node
    if len(args) < 1:
        raise InvalidArguments('Not enough arguments to project(). Needs at least the project name.')
    proj_name = args[0]
    proj_vers = kwargs.get('version', 'undefined')
    proj_langs = self.flatten_args(args[1:])
    if isinstance(proj_vers, ElementaryNode):
        proj_vers = proj_vers.value
    if not isinstance(proj_vers, str):
        proj_vers = 'undefined'
    self.project_data = {'descriptive_name': proj_name, 'version': proj_vers}
    optfile = os.path.join(self.source_root, self.subdir, 'meson.options')
    if not os.path.exists(optfile):
        optfile = os.path.join(self.source_root, self.subdir, 'meson_options.txt')
    if os.path.exists(optfile):
        oi = optinterpreter.OptionInterpreter(self.subproject)
        oi.process(optfile)
        self.coredata.update_project_options(oi.options)
    def_opts = self.flatten_args(kwargs.get('default_options', []))
    _project_default_options = mesonlib.stringlistify(def_opts)
    self.project_default_options = cdata.create_options_dict(_project_default_options, self.subproject)
    self.default_options.update(self.project_default_options)
    self.coredata.set_default_options(self.default_options, self.subproject, self.environment)
    if not self.is_subproject() and 'subproject_dir' in kwargs:
        spdirname = kwargs['subproject_dir']
        if isinstance(spdirname, BaseStringNode):
            assert isinstance(spdirname.value, str)
            self.subproject_dir = spdirname.value
    if not self.is_subproject():
        self.project_data['subprojects'] = []
        subprojects_dir = os.path.join(self.source_root, self.subproject_dir)
        if os.path.isdir(subprojects_dir):
            for i in os.listdir(subprojects_dir):
                if os.path.isdir(os.path.join(subprojects_dir, i)):
                    self.do_subproject(SubProject(i))
    self.coredata.init_backend_options(self.backend)
    options = {k: v for k, v in self.environment.options.items() if k.is_backend()}
    self.coredata.set_options(options)
    self._add_languages(proj_langs, True, MachineChoice.HOST)
    self._add_languages(proj_langs, True, MachineChoice.BUILD)