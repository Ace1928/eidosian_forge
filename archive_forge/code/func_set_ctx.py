import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
def set_ctx(self, ctx: str) -> 'Node':
    """Reset the context of a node and all child nodes.  Per default the
        parser will all generate nodes that have a 'load' context as it's the
        most common one.  This method is used in the parser to set assignment
        targets and other nodes to a store context.
        """
    todo = deque([self])
    while todo:
        node = todo.popleft()
        if 'ctx' in node.fields:
            node.ctx = ctx
        todo.extend(node.iter_child_nodes())
    return self