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
def split_version_string(version: str) -> T.Dict[str, T.Union[str, int]]:
    vers_list = version.split('.')
    return {'full': version, 'major': int(vers_list[0] if len(vers_list) > 0 else 0), 'minor': int(vers_list[1] if len(vers_list) > 1 else 0), 'patch': int(vers_list[2] if len(vers_list) > 2 else 0)}