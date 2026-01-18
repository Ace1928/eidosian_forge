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
def libraries_substitution(self) -> None:
    self.substitutions['lib_dir_declarations'] = '\n'.join([f"lib_dir_{i} = declare_dependency(link_args : ['-L{lib_dir}'])" for i, lib_dir in enumerate(self.library_dirs)])
    self.substitutions['lib_declarations'] = '\n'.join([f"{lib} = declare_dependency(link_args : ['-l{lib}'])" for lib in self.libraries])
    indent = ' ' * 21
    self.substitutions['lib_list'] = f'\n{indent}'.join([f'{indent}{lib},' for lib in self.libraries])
    self.substitutions['lib_dir_list'] = f'\n{indent}'.join([f'{indent}lib_dir_{i},' for i in range(len(self.library_dirs))])