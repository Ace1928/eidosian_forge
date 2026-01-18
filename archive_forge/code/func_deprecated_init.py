import functools
import types
import warnings
import importlib
import sys
from gi import PyGIDeprecationWarning
from gi._gi import CallableInfo, pygobject_new_full
from gi._constants import \
from pkgutil import extend_path
def deprecated_init(super_init_func, arg_names, ignore=tuple(), deprecated_aliases={}, deprecated_defaults={}, category=PyGIDeprecationWarning, stacklevel=2):
    """Wrapper for deprecating GObject based __init__ methods which specify
    defaults already available or non-standard defaults.

    :param callable super_init_func:
        Initializer to wrap.
    :param list arg_names:
        Ordered argument name list.
    :param list ignore:
        List of argument names to ignore when calling the wrapped function.
        This is useful for function which take a non-standard keyword that is munged elsewhere.
    :param dict deprecated_aliases:
        Dictionary mapping a keyword alias to the actual g_object_newv keyword.
    :param dict deprecated_defaults:
        Dictionary of non-standard defaults that will be used when the
        keyword is not explicitly passed.
    :param Exception category:
        Exception category of the error.
    :param int stacklevel:
        Stack level for the deprecation passed on to warnings.warn
    :returns: Wrapped version of ``super_init_func`` which gives a deprecation
        warning when non-keyword args or aliases are used.
    :rtype: callable
    """

    def new_init(self, *args, **kwargs):
        """Initializer for a GObject based classes with support for property
        sets through the use of explicit keyword arguments.
        """
        if args:
            warnings.warn('Using positional arguments with the GObject constructor has been deprecated. Please specify keyword(s) for "%s" or use a class specific constructor. See: https://wiki.gnome.org/PyGObject/InitializerDeprecations' % ', '.join(arg_names[:len(args)]), category, stacklevel=stacklevel)
            new_kwargs = dict(zip(arg_names, args))
        else:
            new_kwargs = {}
        new_kwargs.update(kwargs)
        aliases_used = []
        for key, alias in deprecated_aliases.items():
            if alias in new_kwargs:
                new_kwargs[key] = new_kwargs.pop(alias)
                aliases_used.append(key)
        if aliases_used:
            warnings.warn('The keyword(s) "%s" have been deprecated in favor of "%s" respectively. See: https://wiki.gnome.org/PyGObject/InitializerDeprecations' % (', '.join((deprecated_aliases[k] for k in sorted(aliases_used))), ', '.join(sorted(aliases_used))), category, stacklevel=stacklevel)
        defaults_used = []
        for key, value in deprecated_defaults.items():
            if key not in new_kwargs:
                new_kwargs[key] = deprecated_defaults[key]
                defaults_used.append(key)
        if defaults_used:
            warnings.warn('Initializer is relying on deprecated non-standard defaults. Please update to explicitly use: %s See: https://wiki.gnome.org/PyGObject/InitializerDeprecations' % ', '.join(('%s=%s' % (k, deprecated_defaults[k]) for k in sorted(defaults_used))), category, stacklevel=stacklevel)
        for key in ignore:
            if key in new_kwargs:
                new_kwargs.pop(key)
        return super_init_func(self, **new_kwargs)
    return new_init