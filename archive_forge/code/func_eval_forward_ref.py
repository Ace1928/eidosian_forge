import collections.abc
import io
import itertools
import types
import typing
def eval_forward_ref(ref, forward_refs=None):
    """
    eval forward_refs in all cPython versions
    """
    localns = forward_refs or {}
    if hasattr(typing, '_eval_type'):
        _eval_type = getattr(typing, '_eval_type')
        return _eval_type(ref, globals(), localns)
    if hasattr(ref, '_eval_type'):
        _eval_type = getattr(ref, '_eval_type')
        return _eval_type(globals(), localns)
    raise NotImplementedError()