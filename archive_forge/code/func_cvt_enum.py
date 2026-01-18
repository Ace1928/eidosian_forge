from __future__ import annotations
from collections import namedtuple
def cvt_enum(self, value):
    return self.enum.get(value, value) if self.enum else value