from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
def include_substitution(self) -> None:
    indent = ' ' * 21
    self.substitutions['inc_list'] = f',\n{indent}'.join([f"{indent}'{inc}'" for inc in self.include_dirs])