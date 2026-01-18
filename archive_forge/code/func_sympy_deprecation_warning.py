import warnings
import contextlib
from textwrap import dedent
def sympy_deprecation_warning(message, *, deprecated_since_version, active_deprecations_target, stacklevel=3):
    '''
    Warn that a feature is deprecated in SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    To mark an entire function or class as deprecated, you can use the
    :func:`@deprecated <sympy.utilities.decorator.deprecated>` decorator.

    Parameters
    ==========

    message: str

         The deprecation message. This may span multiple lines and contain
         code examples. Messages should be wrapped to 80 characters. The
         message is automatically dedented and leading and trailing whitespace
         stripped. Messages may include dynamic content based on the user
         input, but avoid using ``str(expression)`` if an expression can be
         arbitrary, as it might be huge and make the warning message
         unreadable.

    deprecated_since_version: str

         The version of SymPy the feature has been deprecated since. For new
         deprecations, this should be the version in `sympy/release.py
         <https://github.com/sympy/sympy/blob/master/sympy/release.py>`_
         without the ``.dev``. If the next SymPy version ends up being
         different from this, the release manager will need to update any
         ``SymPyDeprecationWarning``\\s using the incorrect version. This
         argument is required and must be passed as a keyword argument.
         (example:  ``deprecated_since_version="1.10"``).

    active_deprecations_target: str

        The Sphinx target corresponding to the section for the deprecation in
        the :ref:`active-deprecations` document (see
        ``doc/src/explanation/active-deprecations.md``). This is used to
        automatically generate a URL to the page in the warning message. This
        argument is required and must be passed as a keyword argument.
        (example: ``active_deprecations_target="deprecated-feature-abc"``)

    stacklevel: int (default: 3)

        The ``stacklevel`` parameter that is passed to ``warnings.warn``. If
        you create a wrapper that calls this function, this should be
        increased so that the warning message shows the user line of code that
        produced the warning. Note that in some cases there will be multiple
        possible different user code paths that could result in the warning.
        In that case, just choose the smallest common stacklevel.

    Examples
    ========

    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> def is_this_zero(x, y=0):
    ...     """
    ...     Determine if x = 0.
    ...
    ...     Parameters
    ...     ==========
    ...
    ...     x : Expr
    ...       The expression to check.
    ...
    ...     y : Expr, optional
    ...       If provided, check if x = y.
    ...
    ...       .. deprecated:: 1.1
    ...
    ...          The ``y`` argument to ``is_this_zero`` is deprecated. Use
    ...          ``is_this_zero(x - y)`` instead.
    ...
    ...     """
    ...     from sympy import simplify
    ...
    ...     if y != 0:
    ...         sympy_deprecation_warning("""
    ...     The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.""",
    ...             deprecated_since_version="1.1",
    ...             active_deprecations_target='is-this-zero-y-deprecation')
    ...     return simplify(x - y) == 0
    >>> is_this_zero(0)
    True
    >>> is_this_zero(1, 1) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#is-this-zero-y-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      is_this_zero(1, 1)
    True

    See Also
    ========

    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    w = SymPyDeprecationWarning(message, deprecated_since_version=deprecated_since_version, active_deprecations_target=active_deprecations_target)
    warnings.warn(w, stacklevel=stacklevel)