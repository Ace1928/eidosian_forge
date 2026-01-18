import abc
import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import StartTraceback, find_free_port
from ray.exceptions import RayActorError
from ray.types import ObjectRef
def construct_train_func(train_func: Union[Callable[[], T], Callable[[Dict[str, Any]], T]], config: Optional[Dict[str, Any]], fn_arg_name: Optional[str]='train_func', discard_returns: bool=False) -> Callable[[], T]:
    """Validates and constructs the training function to execute.
    Args:
        train_func: The training function to execute.
            This can either take in no arguments or a ``config`` dict.
        config (Optional[Dict]): Configurations to pass into
            ``train_func``. If None then an empty Dict will be created.
        fn_arg_name (Optional[str]): The name of training function to use for error
            messages.
        discard_returns: Whether to discard any returns from train_func or not.
    Returns:
        A valid training function.
    Raises:
        ValueError: if the input ``train_func`` is invalid.
    """
    signature = inspect.signature(train_func)
    num_params = len(signature.parameters)
    if discard_returns:

        @functools.wraps(train_func)
        def discard_return_wrapper(*args, **kwargs):
            try:
                train_func(*args, **kwargs)
            except Exception as e:
                raise StartTraceback from e
        wrapped_train_func = discard_return_wrapper
    else:
        wrapped_train_func = train_func
    if num_params > 1:
        err_msg = f'{fn_arg_name} should take in 0 or 1 arguments, but it accepts {num_params} arguments instead.'
        raise ValueError(err_msg)
    elif num_params == 1:
        config = {} if config is None else config

        @functools.wraps(wrapped_train_func)
        def train_fn():
            try:
                return wrapped_train_func(config)
            except Exception as e:
                raise StartTraceback from e
    else:

        @functools.wraps(wrapped_train_func)
        def train_fn():
            try:
                return wrapped_train_func()
            except Exception as e:
                raise StartTraceback from e
    return train_fn