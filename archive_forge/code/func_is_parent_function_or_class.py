import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_parent_function_or_class(node: Node) -> bool:
    assert node.type in {syms.suite, syms.simple_stmt}
    assert node.parent is not None
    return node.parent.type in {syms.funcdef, syms.classdef}