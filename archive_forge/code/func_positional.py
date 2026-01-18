import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def positional(max_positional_args):
    """A decorator to declare that only the first N arguments my be positional.

    This decorator makes it easy to support Python 3 style keyword-only
    parameters. For example, in Python 3 it is possible to write::

        def fn(pos1, *, kwonly1=None, kwonly1=None):
            ...

    All named parameters after ``*`` must be a keyword::

        fn(10, 'kw1', 'kw2')  # Raises exception.
        fn(10, kwonly1='kw1')  # Ok.

    Example
    ^^^^^^^

    To define a function like above, do::

        @positional(1)
        def fn(pos1, kwonly1=None, kwonly2=None):
            ...

    If no default value is provided to a keyword argument, it becomes a
    required keyword argument::

        @positional(0)
        def fn(required_kw):
            ...

    This must be called with the keyword parameter::

        fn()  # Raises exception.
        fn(10)  # Raises exception.
        fn(required_kw=10)  # Ok.

    When defining instance or class methods always remember to account for
    ``self`` and ``cls``::

        class MyClass(object):

            @positional(2)
            def my_method(self, pos1, kwonly1=None):
                ...

            @classmethod
            @positional(2)
            def my_method(cls, pos1, kwonly1=None):
                ...

    The positional decorator behavior is controlled by
    ``_helpers.positional_parameters_enforcement``, which may be set to
    ``POSITIONAL_EXCEPTION``, ``POSITIONAL_WARNING`` or
    ``POSITIONAL_IGNORE`` to raise an exception, log a warning, or do
    nothing, respectively, if a declaration is violated.

    Args:
        max_positional_arguments: Maximum number of positional arguments. All
                                  parameters after the this index must be
                                  keyword only.

    Returns:
        A decorator that prevents using arguments after max_positional_args
        from being used as positional parameters.

    Raises:
        TypeError: if a key-word only argument is provided as a positional
                   parameter, but only if
                   _helpers.positional_parameters_enforcement is set to
                   POSITIONAL_EXCEPTION.
    """

    def positional_decorator(wrapped):

        @functools.wraps(wrapped)
        def positional_wrapper(*args, **kwargs):
            if len(args) > max_positional_args:
                plural_s = ''
                if max_positional_args != 1:
                    plural_s = 's'
                message = '{function}() takes at most {args_max} positional argument{plural} ({args_given} given)'.format(function=wrapped.__name__, args_max=max_positional_args, args_given=len(args), plural=plural_s)
                if positional_parameters_enforcement == POSITIONAL_EXCEPTION:
                    raise TypeError(message)
                elif positional_parameters_enforcement == POSITIONAL_WARNING:
                    logger.warning(message)
            return wrapped(*args, **kwargs)
        return positional_wrapper
    if isinstance(max_positional_args, six.integer_types):
        return positional_decorator
    else:
        args, _, _, defaults = inspect.getargspec(max_positional_args)
        return positional(len(args) - len(defaults))(max_positional_args)