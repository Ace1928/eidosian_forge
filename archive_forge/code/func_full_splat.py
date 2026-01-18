import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def full_splat(self, args: List) -> str:
    args_str = ''.join((str(arg) for arg in args))
    return f'[*]{args_str}'