from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def close_bf(self) -> None:
    if self.bf is not None:
        if self.bf_perms is not None:
            os.chmod(self.bf.fileno(), self.bf_perms)
            self.bf_perms = None
        self.bf.close()
        self.bf = None