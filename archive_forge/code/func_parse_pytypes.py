from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def parse_pytypes(s):
    fake_def = '#pythran export fake({})'.format(s)
    specs = SpecParser()(fake_def)
    return specs.functions['fake'][0]