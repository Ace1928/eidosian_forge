import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
def args_as_const(node: t.Union['_FilterTestCommon', 'Call'], eval_ctx: t.Optional[EvalContext]) -> t.Tuple[t.List[t.Any], t.Dict[t.Any, t.Any]]:
    args = [x.as_const(eval_ctx) for x in node.args]
    kwargs = dict((x.as_const(eval_ctx) for x in node.kwargs))
    if node.dyn_args is not None:
        try:
            args.extend(node.dyn_args.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e
    if node.dyn_kwargs is not None:
        try:
            kwargs.update(node.dyn_kwargs.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e
    return (args, kwargs)