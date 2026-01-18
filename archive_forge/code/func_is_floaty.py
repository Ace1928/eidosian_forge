import re
from lib2to3.fixer_util import Leaf, Node, Comma
from lib2to3 import fixer_base
from libfuturize.fixer_util import (token, future_import, touch_import_top,
def is_floaty(node):
    return _is_floaty(node.prev_sibling) or _is_floaty(node.next_sibling)