from .sage_helper import _within_sage
from .pari import *
import re
def pari_type(self):
    return self.gen.type()