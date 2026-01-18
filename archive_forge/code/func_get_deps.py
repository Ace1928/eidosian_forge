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
def get_deps(self) -> T.Generator[str, None, None]:
    sec = self.find_section(b'.dynstr')
    for i in self.dynamic:
        if i.d_tag == DT_NEEDED:
            offset = sec.sh_offset + i.val
            self.bf.seek(offset)
            yield self.read_str().decode()