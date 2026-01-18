from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def parse_dynamic(self) -> None:
    sec = self.find_section(b'.dynamic')
    if sec is None:
        return
    self.bf.seek(sec.sh_offset)
    while True:
        e = DynamicEntry(self.bf, self.ptrsize, self.is_le)
        self.dynamic.append(e)
        if e.d_tag == 0:
            break