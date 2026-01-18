from contextlib import contextmanager
from typing import Generator
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
@contextmanager
def isolate_rng(include_cuda: bool=True) -> Generator[None, None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.

    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.

    Args:
        include_cuda: Whether to allow this function to also control the `torch.cuda` random number generator.
            Set this to ``False`` when using the function in a forked process where CUDA re-initialization is
            prohibited.

    Example:
        >>> import torch
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1) for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)
        tensor([0.7576])

    """
    states = _collect_rng_states(include_cuda)
    yield
    _set_rng_states(states)