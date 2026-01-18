from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def read_str(self) -> bytes:
    arr = []
    x = self.bf.read(1)
    while x != b'\x00':
        arr.append(x)
        x = self.bf.read(1)
        if x == b'':
            raise RuntimeError('Tried to read past the end of the file')
    return b''.join(arr)