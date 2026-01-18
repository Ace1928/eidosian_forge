import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def has_sunken_brackets(tokens: List[tokenize.TokenInfo]):
    """Check if the depth of brackets in the list of tokens drops below 0"""
    parenlev = 0
    for token in tokens:
        if token.string in {'(', '[', '{'}:
            parenlev += 1
        elif token.string in {')', ']', '}'}:
            parenlev -= 1
            if parenlev < 0:
                return True
    return False