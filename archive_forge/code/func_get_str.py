import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def get_str(self, leadchar):
    if self.contents == b'\n' and leadchar == b' ' and False:
        return b'\n'
    if not self.contents.endswith(b'\n'):
        terminator = b'\n' + NO_NL
    else:
        terminator = b''
    return leadchar + self.contents + terminator