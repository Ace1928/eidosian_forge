from collections.abc import Sequence
import functools
from autograd import numpy as _np
from .tensor import tensor
def extract_tensors(x):
    """Iterate through an iterable, and extract any PennyLane
    tensors that appear.

    Args:
        x (.tensor or Sequence): an input tensor or sequence

    Yields:
        tensor: the next tensor in the sequence. If the input was a single
        tensor, than the tensor is yielded and the iterator completes.

    **Example**

    >>> from pennylane import numpy as np
    >>> import numpy as onp
    >>> iterator = np.extract_tensors([0.1, np.array(0.1), "string", onp.array(0.5)])
    >>> list(iterator)
    [tensor(0.1, requires_grad=True)]
    """
    if isinstance(x, tensor):
        yield x
    elif isinstance(x, Sequence) and (not isinstance(x, (str, bytes))):
        for item in x:
            yield from extract_tensors(item)