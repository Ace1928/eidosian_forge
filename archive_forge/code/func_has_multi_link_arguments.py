from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def has_multi_link_arguments(self, args: T.List[str], env: 'Environment') -> T.Tuple[bool, bool]:
    return self._has_multi_link_arguments(args, env, 'int main(void) { return 0; }\n')