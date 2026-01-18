import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def expr_term(self, args: List) -> Any:
    args = self.strip_new_line_tokens(args)
    if args[0] == 'true':
        return True
    if args[0] == 'false':
        return False
    if args[0] == 'null':
        return None
    if args[0] == '(':
        return args[1]
    return args[0]