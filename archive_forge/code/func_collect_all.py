from __future__ import annotations
from typing import cast, Callable, Generic, List, Optional, Type, TypeVar, Union
import torch
def collect_all(futures: List[Future]) -> Future[List[Future]]:
    """
    Collects the provided :class:`~torch.futures.Future` objects into a single
    combined :class:`~torch.futures.Future` that is completed when all of the
    sub-futures are completed.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` objects.

    Returns:
        Returns a :class:`~torch.futures.Future` object to a list of the passed
        in Futures.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
        >>> fut0 = torch.futures.Future()
        >>> fut1 = torch.futures.Future()
        >>> fut = torch.futures.collect_all([fut0, fut1])
        >>> fut0.set_result(0)
        >>> fut1.set_result(1)
        >>> fut_list = fut.wait()
        >>> print(f"fut0 result = {fut_list[0].wait()}")
        fut0 result = 0
        >>> print(f"fut1 result = {fut_list[1].wait()}")
        fut1 result = 1
    """
    return cast(Future[List[Future]], torch._C._collect_all(cast(List[torch._C.Future], futures)))