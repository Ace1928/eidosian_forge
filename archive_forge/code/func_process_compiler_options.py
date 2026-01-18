from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
def process_compiler_options(self, lang: str, comp: Compiler, env: Environment, subproject: str) -> None:
    from . import compilers
    self.add_compiler_options(comp.get_options(), lang, comp.for_machine, env, subproject)
    enabled_opts: T.List[OptionKey] = []
    for key in comp.base_options:
        if subproject:
            skey = key.evolve(subproject=subproject)
        else:
            skey = key
        if skey not in self.options:
            self.options[skey] = copy.deepcopy(compilers.base_options[key])
            if skey in env.options:
                self.options[skey].set_value(env.options[skey])
                enabled_opts.append(skey)
            elif subproject and key in env.options:
                self.options[skey].set_value(env.options[key])
                enabled_opts.append(skey)
            if subproject and key not in self.options:
                self.options[key] = copy.deepcopy(self.options[skey])
        elif skey in env.options:
            self.options[skey].set_value(env.options[skey])
        elif subproject and key in env.options:
            self.options[skey].set_value(env.options[key])
    self.emit_base_options_warnings(enabled_opts)