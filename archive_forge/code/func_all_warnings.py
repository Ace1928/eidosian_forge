from contextlib import contextmanager
import sys
import warnings
import re
import functools
import os
@contextmanager
def all_warnings():
    """
    Context for use in testing to ensure that all warnings are raised.

    Examples
    --------
    >>> import warnings
    >>> def foo():
    ...     warnings.warn(RuntimeWarning("bar"), stacklevel=2)

    We raise the warning once, while the warning filter is set to "once".
    Hereafter, the warning is invisible, even with custom filters:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('once')
    ...     foo()                         # doctest: +SKIP

    We can now run ``foo()`` without a warning being raised:

    >>> from numpy.testing import assert_warns
    >>> foo()                             # doctest: +SKIP

    To catch the warning, we call in the help of ``all_warnings``:

    >>> with all_warnings():
    ...     assert_warns(RuntimeWarning, foo)
    """
    import inspect
    frame = inspect.currentframe()
    if frame:
        for f in inspect.getouterframes(frame):
            f[0].f_locals['__warningregistry__'] = {}
    del frame
    for mod_name, mod in list(sys.modules.items()):
        try:
            mod.__warningregistry__.clear()
        except AttributeError:
            pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield w