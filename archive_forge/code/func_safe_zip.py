from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, overload
def safe_zip(*args):
    """Strict zip that requires all arguments to be the same length."""
    seqs = [arg if isinstance(arg, Sequence) else list(arg) for arg in args]
    if len(set(map(len, seqs))) > 1:
        raise ValueError(f'length mismatch: {list(map(len, seqs))}')
    return zip(*seqs)