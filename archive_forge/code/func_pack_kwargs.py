from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite)

    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return (tuple(kwarg_keys), tuple(flat_args))