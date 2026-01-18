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
def get_test_list(testdata: T.List[backends.TestSerialisation]) -> T.List[T.Dict[str, T.Union[str, int, T.List[str], T.Dict[str, str]]]]:
    result: T.List[T.Dict[str, T.Union[str, int, T.List[str], T.Dict[str, str]]]] = []
    for t in testdata:
        to: T.Dict[str, T.Union[str, int, T.List[str], T.Dict[str, str]]] = {}
        if isinstance(t.fname, str):
            fname = [t.fname]
        else:
            fname = t.fname
        to['cmd'] = fname + t.cmd_args
        if isinstance(t.env, mesonlib.EnvironmentVariables):
            to['env'] = t.env.get_env({})
        else:
            to['env'] = t.env
        to['name'] = t.name
        to['workdir'] = t.workdir
        to['timeout'] = t.timeout
        to['suite'] = t.suite
        to['is_parallel'] = t.is_parallel
        to['priority'] = t.priority
        to['protocol'] = str(t.protocol)
        to['depends'] = t.depends
        to['extra_paths'] = t.extra_paths
        result.append(to)
    return result