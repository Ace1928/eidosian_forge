import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def object_elem(self, args: List) -> Dict:
    key = self.strip_quotes(args[0])
    value = self.to_string_dollar(args[1])
    return {key: value}