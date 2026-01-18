import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def int_lit(self, args: List) -> int:
    return int(''.join([str(arg) for arg in args]))