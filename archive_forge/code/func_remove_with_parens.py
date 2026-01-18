import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def remove_with_parens(node: Node, parent: Node) -> None:
    """Recursively hide optional parens in `with` statements."""
    if node.type == syms.atom:
        if maybe_make_parens_invisible_in_atom(node, parent=parent, remove_brackets_around_comma=True):
            wrap_in_parentheses(parent, node, visible=False)
        if isinstance(node.children[1], Node):
            remove_with_parens(node.children[1], node)
    elif node.type == syms.testlist_gexp:
        for child in node.children:
            if isinstance(child, Node):
                remove_with_parens(child, node)
    elif node.type == syms.asexpr_test and (not any((leaf.type == token.COLONEQUAL for leaf in node.leaves()))):
        if maybe_make_parens_invisible_in_atom(node.children[0], parent=node, remove_brackets_around_comma=True):
            wrap_in_parentheses(node, node.children[0], visible=False)