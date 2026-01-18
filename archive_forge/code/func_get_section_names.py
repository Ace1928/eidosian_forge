from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
@generate_list
def get_section_names(self) -> T.Generator[str, None, None]:
    section_names = self.sections[self.e_shstrndx]
    for i in self.sections:
        self.bf.seek(section_names.sh_offset + i.sh_name)
        yield self.read_str().decode()