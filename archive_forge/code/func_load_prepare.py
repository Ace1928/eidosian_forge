from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def load_prepare(self):
    if not self.im or self.im.mode != self.mode or self.im.size != self.size:
        self.im = Image.core.new(self.mode, self.size)
    if self.mode == 'P':
        Image.Image.load(self)