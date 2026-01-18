import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def for_intro(self, args: List) -> str:
    args = self.strip_new_line_tokens(args)
    return ' '.join([str(arg) for arg in args])