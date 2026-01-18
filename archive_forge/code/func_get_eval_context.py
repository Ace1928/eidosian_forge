import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
def get_eval_context(node: 'Node', ctx: t.Optional[EvalContext]) -> EvalContext:
    if ctx is None:
        if node.environment is None:
            raise RuntimeError('if no eval context is passed, the node must have an attached environment.')
        return EvalContext(node.environment)
    return ctx