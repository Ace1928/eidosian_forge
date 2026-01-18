from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def list_projinfo(builddata: build.Build) -> T.Dict[str, T.Union[str, T.List[T.Dict[str, str]]]]:
    result: T.Dict[str, T.Union[str, T.List[T.Dict[str, str]]]] = {'version': builddata.project_version, 'descriptive_name': builddata.project_name, 'subproject_dir': builddata.subproject_dir}
    subprojects = []
    for k, v in builddata.subprojects.items():
        c: T.Dict[str, str] = {'name': k, 'version': v, 'descriptive_name': builddata.projects.get(k)}
        subprojects.append(c)
    result['subprojects'] = subprojects
    return result