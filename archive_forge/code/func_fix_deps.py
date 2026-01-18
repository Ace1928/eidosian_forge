from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def fix_deps(self, prefix: bytes) -> None:
    sec = self.find_section(b'.dynstr')
    deps = []
    for i in self.dynamic:
        if i.d_tag == DT_NEEDED:
            deps.append(i)
    for i in deps:
        offset = sec.sh_offset + i.val
        self.bf.seek(offset)
        name = self.read_str()
        if name.startswith(prefix):
            basename = name.rsplit(b'/', maxsplit=1)[-1]
            padding = b'\x00' * (len(name) - len(basename))
            newname = basename + padding
            assert len(newname) == len(name)
            self.bf.seek(offset)
            self.bf.write(newname)