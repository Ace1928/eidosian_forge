import os
import sys
import stat
import fnmatch
import collections
import errno
def register_unpack_format(name, extensions, function, extra_args=None, description=''):
    """Registers an unpack format.

    `name` is the name of the format. `extensions` is a list of extensions
    corresponding to the format.

    `function` is the callable that will be
    used to unpack archives. The callable will receive archives to unpack.
    If it's unable to handle an archive, it needs to raise a ReadError
    exception.

    If provided, `extra_args` is a sequence of
    (name, value) tuples that will be passed as arguments to the callable.
    description can be provided to describe the format, and will be returned
    by the get_unpack_formats() function.
    """
    if extra_args is None:
        extra_args = []
    _check_unpack_options(extensions, function, extra_args)
    _UNPACK_FORMATS[name] = (extensions, function, extra_args, description)