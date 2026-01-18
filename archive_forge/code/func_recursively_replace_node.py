from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def recursively_replace_node(tree, old_node, new_node):
    replace_in = RecursiveNodeReplacer(old_node, new_node)
    replace_in(tree)