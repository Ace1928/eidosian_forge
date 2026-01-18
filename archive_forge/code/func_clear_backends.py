import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def clear_backends(domain, registered=True, globals=False):
    """
    This utility method clears registered backends.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves.

    .. warning::
        Do NOT use this method inside a multimethod call, or the
        program is likely to crash.

    Parameters
    ----------
    domain : Optional[str]
        The domain for which to de-register backends. ``None`` means
        de-register for all domains.
    registered : bool
        Whether or not to clear registered backends. See :obj:`register_backend`.
    globals : bool
        Whether or not to clear global backends. See :obj:`set_global_backend`.

    See Also
    --------
    register_backend : Register a backend globally.
    set_global_backend : Set a global backend.
    """
    _uarray.clear_backends(domain, registered, globals)