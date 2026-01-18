import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_choice(self, f, indent):
    space_left = self.max_width - self.pos
    if space_left > 0 and fits(f.children[0], space_left):
        self.pp(f.children[0], indent)
    else:
        self.pp(f.children[1], indent)