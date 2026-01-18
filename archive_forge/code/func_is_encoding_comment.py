from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def is_encoding_comment(node):
    """
    Comments are prefixes for Leaf nodes. Returns whether the given node has a
    prefix that looks like an encoding line:

        # coding: utf-8
        # encoding: utf-8
        # -*- coding: <encoding name> -*-
        # vim: set fileencoding=<encoding name> :
    """
    return bool(re.match(ENCODING_REGEX, node.prefix))