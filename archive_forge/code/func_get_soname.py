from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def get_soname(self) -> T.Optional[str]:
    soname = None
    strtab = None
    for i in self.dynamic:
        if i.d_tag == DT_SONAME:
            soname = i
        if i.d_tag == DT_STRTAB:
            strtab = i
    if soname is None or strtab is None:
        return None
    self.bf.seek(strtab.val + soname.val)
    return self.read_str().decode()