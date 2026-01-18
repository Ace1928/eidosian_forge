import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import xxhash
from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
def fingerprint_transform(inplace: bool, use_kwargs: Optional[List[str]]=None, ignore_kwargs: Optional[List[str]]=None, fingerprint_names: Optional[List[str]]=None, randomized_function: bool=False, version: Optional[str]=None):
    """
    Wrapper for dataset transforms to update the dataset fingerprint using ``update_fingerprint``
    Args:
        inplace (:obj:`bool`):  If inplace is True, the fingerprint of the dataset is updated inplace.
            Otherwise, a parameter "new_fingerprint" is passed to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset.
        use_kwargs (:obj:`List[str]`, optional): optional white list of argument names to take into account
            to update the fingerprint to the wrapped method that should take care of
            setting the fingerprint of the returned Dataset. By default all the arguments are used.
        ignore_kwargs (:obj:`List[str]`, optional): optional black list of argument names to take into account
            to update the fingerprint. Note that ignore_kwargs prevails on use_kwargs.
        fingerprint_names (:obj:`List[str]`, optional, defaults to ["new_fingerprint"]):
            If the dataset transforms is not inplace and returns a DatasetDict, then it can require
            several fingerprints (one per dataset in the DatasetDict). By specifying fingerprint_names,
            one fingerprint named after each element of fingerprint_names is going to be passed.
        randomized_function (:obj:`bool`, defaults to False): If the dataset transform is random and has
            optional parameters "seed" and "generator", then you can set randomized_function to True.
            This way, even if users set "seed" and "generator" to None, then the fingerprint is
            going to be randomly generated depending on numpy's current state. In this case, the
            generator is set to np.random.default_rng(np.random.get_state()[1][0]).
        version (:obj:`str`, optional): version of the transform. The version is taken into account when
            computing the fingerprint. If a datase transform changes (or at least if the output data
            that are cached changes), then one should increase the version. If the version stays the
            same, then old cached data could be reused that are not compatible with the new transform.
            It should be in the format "MAJOR.MINOR.PATCH".
    """
    if use_kwargs is not None and (not isinstance(use_kwargs, list)):
        raise ValueError(f'use_kwargs is supposed to be a list, not {type(use_kwargs)}')
    if ignore_kwargs is not None and (not isinstance(ignore_kwargs, list)):
        raise ValueError(f'ignore_kwargs is supposed to be a list, not {type(use_kwargs)}')
    if inplace and fingerprint_names:
        raise ValueError('fingerprint_names are only used when inplace is False')
    fingerprint_names = fingerprint_names if fingerprint_names is not None else ['new_fingerprint']

    def _fingerprint(func):
        if not inplace and (not all((name in func.__code__.co_varnames for name in fingerprint_names))):
            raise ValueError(f'function {func} is missing parameters {fingerprint_names} in signature')
        if randomized_function:
            if 'seed' not in func.__code__.co_varnames:
                raise ValueError(f"'seed' must be in {func}'s signature")
            if 'generator' not in func.__code__.co_varnames:
                raise ValueError(f"'generator' must be in {func}'s signature")
        transform = format_transform_for_fingerprint(func, version=version)

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(func, args, kwargs, use_kwargs=use_kwargs, ignore_kwargs=ignore_kwargs, randomized_function=randomized_function)
            if args:
                dataset: Dataset = args[0]
                args = args[1:]
            else:
                dataset: Dataset = kwargs.pop(next(iter(inspect.signature(func).parameters)))
            if inplace:
                new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
            else:
                for fingerprint_name in fingerprint_names:
                    if kwargs.get(fingerprint_name) is None:
                        kwargs_for_fingerprint['fingerprint_name'] = fingerprint_name
                        kwargs[fingerprint_name] = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
                    else:
                        validate_fingerprint(kwargs[fingerprint_name])
            out = func(dataset, *args, **kwargs)
            if inplace:
                dataset._fingerprint = new_fingerprint
            return out
        wrapper._decorator_name_ = 'fingerprint'
        return wrapper
    return _fingerprint