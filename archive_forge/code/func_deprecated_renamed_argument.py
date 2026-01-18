import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
def deprecated_renamed_argument(old_name, new_name, since, arg_in_kwargs=False, relax=False, pending=False, warning_type=DeprecationWarning, alternative='', message=''):
    """Deprecate a _renamed_ or _removed_ function argument.

    The decorator assumes that the argument with the ``old_name`` was removed
    from the function signature and the ``new_name`` replaced it at the
    **same position** in the signature.  If the ``old_name`` argument is
    given when calling the decorated function the decorator will catch it and
    issue a deprecation warning and pass it on as ``new_name`` argument.

    Parameters
    ----------
    old_name : str or sequence of str
        The old name of the argument.

    new_name : str or sequence of str or None
        The new name of the argument. Set this to `None` to remove the
        argument ``old_name`` instead of renaming it.

    since : str or number or sequence of str or number
        The release at which the old argument became deprecated.

    arg_in_kwargs : bool or sequence of bool, optional
        If the argument is not a named argument (for example it
        was meant to be consumed by ``**kwargs``) set this to
        ``True``.  Otherwise the decorator will throw an Exception
        if the ``new_name`` cannot be found in the signature of
        the decorated function.
        Default is ``False``.

    relax : bool or sequence of bool, optional
        If ``False`` a ``TypeError`` is raised if both ``new_name`` and
        ``old_name`` are given.  If ``True`` the value for ``new_name`` is used
        and a Warning is issued.
        Default is ``False``.

    pending : bool or sequence of bool, optional
        If ``True`` this will hide the deprecation warning and ignore the
        corresponding ``relax`` parameter value.
        Default is ``False``.

    warning_type : Warning
        Warning to be issued.
        Default is `~astropy.utils.exceptions.AstropyDeprecationWarning`.

    alternative : str, optional
        An alternative function or class name that the user may use in
        place of the deprecated object if ``new_name`` is None. The deprecation
        warning will tell the user about this alternative if provided.

    message : str, optional
        A custom warning message. If provided then ``since`` and
        ``alternative`` options will have no effect.

    Raises
    ------
    TypeError
        If the new argument name cannot be found in the function
        signature and arg_in_kwargs was False or if it is used to
        deprecate the name of the ``*args``-, ``**kwargs``-like arguments.
        At runtime such an Error is raised if both the new_name
        and old_name were specified when calling the function and
        "relax=False".

    Notes
    -----
    The decorator should be applied to a function where the **name**
    of an argument was changed but it applies the same logic.

    .. warning::
        If ``old_name`` is a list or tuple the ``new_name`` and ``since`` must
        also be a list or tuple with the same number of entries. ``relax`` and
        ``arg_in_kwarg`` can be a single bool (applied to all) or also a
        list/tuple with the same number of entries like ``new_name``, etc.

    Examples
    --------
    The deprecation warnings are not shown in the following examples.

    To deprecate a positional or keyword argument::

        >>> from astropy.utils.decorators import deprecated_renamed_argument
        >>> @deprecated_renamed_argument('sig', 'sigma', '1.0')
        ... def test(sigma):
        ...     return sigma

        >>> test(2)
        2
        >>> test(sigma=2)
        2
        >>> test(sig=2)  # doctest: +SKIP
        2

    To deprecate an argument caught inside the ``**kwargs`` the
    ``arg_in_kwargs`` has to be set::

        >>> @deprecated_renamed_argument('sig', 'sigma', '1.0',
        ...                             arg_in_kwargs=True)
        ... def test(**kwargs):
        ...     return kwargs['sigma']

        >>> test(sigma=2)
        2
        >>> test(sig=2)  # doctest: +SKIP
        2

    By default providing the new and old keyword will lead to an Exception. If
    a Warning is desired set the ``relax`` argument::

        >>> @deprecated_renamed_argument('sig', 'sigma', '1.0', relax=True)
        ... def test(sigma):
        ...     return sigma

        >>> test(sig=2)  # doctest: +SKIP
        2

    It is also possible to replace multiple arguments. The ``old_name``,
    ``new_name`` and ``since`` have to be `tuple` or `list` and contain the
    same number of entries::

        >>> @deprecated_renamed_argument(['a', 'b'], ['alpha', 'beta'],
        ...                              ['1.0', 1.2])
        ... def test(alpha, beta):
        ...     return alpha, beta

        >>> test(a=2, b=3)  # doctest: +SKIP
        (2, 3)

    In this case ``arg_in_kwargs`` and ``relax`` can be a single value (which
    is applied to all renamed arguments) or must also be a `tuple` or `list`
    with values for each of the arguments.

    """
    cls_iter = (list, tuple)
    if isinstance(old_name, cls_iter):
        n = len(old_name)
        if not isinstance(arg_in_kwargs, cls_iter):
            arg_in_kwargs = [arg_in_kwargs] * n
        if not isinstance(relax, cls_iter):
            relax = [relax] * n
        if not isinstance(pending, cls_iter):
            pending = [pending] * n
        if not isinstance(message, cls_iter):
            message = [message] * n
    else:
        n = 1
        old_name = [old_name]
        new_name = [new_name]
        since = [since]
        arg_in_kwargs = [arg_in_kwargs]
        relax = [relax]
        pending = [pending]
        message = [message]

    def decorator(function):
        arguments = signature(function).parameters
        keys = list(arguments.keys())
        position = [None] * n
        for i in range(n):
            if arg_in_kwargs[i]:
                pass
            else:
                if new_name[i] is None:
                    param = arguments[old_name[i]]
                elif new_name[i] in arguments:
                    param = arguments[new_name[i]]
                else:
                    raise TypeError(f'"{new_name[i]}" was not specified in the function signature. If it was meant to be part of "**kwargs" then set "arg_in_kwargs" to "True"')
                if param.kind == param.POSITIONAL_OR_KEYWORD:
                    if new_name[i] is None:
                        position[i] = keys.index(old_name[i])
                    else:
                        position[i] = keys.index(new_name[i])
                elif param.kind == param.KEYWORD_ONLY:
                    position[i] = None
                else:
                    raise TypeError(f'cannot replace argument "{new_name[i]}" of kind {repr(param.kind)}.')

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for i in range(n):
                msg = message[i] or f'"{old_name[i]}" was deprecated in version {since[i]} and will be removed in a future version. '
                if old_name[i] in kwargs:
                    value = kwargs.pop(old_name[i])
                    if not pending[i]:
                        if not message[i]:
                            if new_name[i] is not None:
                                msg += f'Use argument "{new_name[i]}" instead.'
                            elif alternative:
                                msg += f'\n        Use {alternative} instead.'
                        warnings.warn(msg, warning_type, stacklevel=2)
                    newarg_in_args = position[i] is not None and len(args) > position[i]
                    newarg_in_kwargs = new_name[i] in kwargs
                    if newarg_in_args or newarg_in_kwargs:
                        if not pending[i]:
                            if relax[i]:
                                warnings.warn(f'"{old_name[i]}" and "{new_name[i]}" keywords were set. Using the value of "{new_name[i]}".', UserWarning)
                            else:
                                raise TypeError(f'cannot specify both "{old_name[i]}" and "{new_name[i]}".')
                    elif new_name[i] is not None:
                        kwargs[new_name[i]] = value
                    else:
                        kwargs[old_name[i]] = value
                elif not pending[i] and (not new_name[i]) and position[i] and (len(args) > position[i]):
                    if alternative and (not message[i]):
                        msg += f'\n        Use {alternative} instead.'
                    warnings.warn(msg, warning_type, stacklevel=2)
            return function(*args, **kwargs)
        return wrapper
    return decorator