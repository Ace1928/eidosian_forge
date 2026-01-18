import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def do_token_transforms(self, lines):
    for _ in range(TRANSFORM_LOOP_LIMIT):
        changed, lines = self.do_one_token_transform(lines)
        if not changed:
            return lines
    raise RuntimeError('Input transformation still changing after %d iterations. Aborting.' % TRANSFORM_LOOP_LIMIT)