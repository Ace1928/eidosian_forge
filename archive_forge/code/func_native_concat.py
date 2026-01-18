import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template
def native_concat(values: t.Iterable[t.Any]) -> t.Optional[t.Any]:
    """Return a native Python type from the list of compiled nodes. If
    the result is a single node, its value is returned. Otherwise, the
    nodes are concatenated as strings. If the result can be parsed with
    :func:`ast.literal_eval`, the parsed value is returned. Otherwise,
    the string is returned.

    :param values: Iterable of outputs to concatenate.
    """
    head = list(islice(values, 2))
    if not head:
        return None
    if len(head) == 1:
        raw = head[0]
        if not isinstance(raw, str):
            return raw
    else:
        if isinstance(values, GeneratorType):
            values = chain(head, values)
        raw = ''.join([str(v) for v in values])
    try:
        return literal_eval(parse(raw, mode='eval'))
    except (ValueError, SyntaxError, MemoryError):
        return raw