from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def detect_vs_dep_prefix(self, tempfilename):
    """VS writes its dependency in a locale dependent format.
        Detect the search prefix to use."""
    for compiler in self.environment.coredata.compilers.host.values():
        if compiler.language in {'fortran', 'masm'}:
            continue
        if compiler.id == 'pgi' and mesonlib.is_windows():
            return open(tempfilename, 'a', encoding='utf-8')
        if compiler.get_argument_syntax() == 'msvc':
            break
    else:
        return open(tempfilename, 'a', encoding='utf-8')
    filebase = 'incdetect.' + compilers.lang_suffixes[compiler.language][0]
    filename = os.path.join(self.environment.get_scratch_dir(), filebase)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(dedent('                #include<stdio.h>\n                int dummy;\n            '))
    pc = subprocess.Popen(compiler.get_exelist() + ['/showIncludes', '/c', filebase], cwd=self.environment.get_scratch_dir(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = pc.communicate()
    matchre = re.compile(b'^(.*\\s)([a-zA-Z]:[\\\\/]|[\\\\\\/]).*stdio.h$')

    def detect_prefix(out):
        for line in re.split(b'\\r?\\n', out):
            match = matchre.match(line)
            if match:
                with open(tempfilename, 'ab') as binfile:
                    binfile.write(b'msvc_deps_prefix = ' + match.group(1) + b'\n')
                return open(tempfilename, 'a', encoding='utf-8')
        return None
    result = detect_prefix(stdout) or detect_prefix(stderr)
    if result:
        return result
    raise MesonException(f'Could not determine vs dep dependency prefix string. output: {stderr} {stdout}')