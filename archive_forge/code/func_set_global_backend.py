import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def set_global_backend(backend, coerce=False, only=False, *, try_last=False):
    """
    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.

    Note that this method is not thread-safe.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves, or by a reference
        implementation, if one exists.

    Parameters
    ----------
    backend
        The backend to register.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.

    See Also
    --------
    set_backend: A context manager that allows setting of backends.
    skip_backend: A context manager that allows skipping of backends.
    """
    _uarray.set_global_backend(backend, coerce, only, try_last)