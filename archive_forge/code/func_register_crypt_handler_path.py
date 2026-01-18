import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def register_crypt_handler_path(name, path):
    """register location to lazy-load handler when requested.

    custom hashes may be registered via :func:`register_crypt_handler`,
    or they may be registered by this function,
    which will delay actually importing and loading the handler
    until a call to :func:`get_crypt_handler` is made for the specified name.

    :arg name: name of handler
    :arg path: module import path

    the specified module path should contain a password hash handler
    called :samp:`{name}`, or the path may contain a colon,
    specifying the module and module attribute to use.
    for example, the following would cause ``get_handler("myhash")`` to look
    for a class named ``myhash`` within the ``myapp.helpers`` module::

        >>> from passlib.registry import registry_crypt_handler_path
        >>> registry_crypt_handler_path("myhash", "myapp.helpers")

    ...while this form would cause ``get_handler("myhash")`` to look
    for a class name ``MyHash`` within the ``myapp.helpers`` module::

        >>> from passlib.registry import registry_crypt_handler_path
        >>> registry_crypt_handler_path("myhash", "myapp.helpers:MyHash")
    """
    _validate_handler_name(name)
    if path.startswith('.'):
        raise ValueError("path cannot start with '.'")
    if ':' in path:
        if path.count(':') > 1:
            raise ValueError("path cannot have more than one ':'")
        if path.find('.', path.index(':')) > -1:
            raise ValueError("path cannot have '.' to right of ':'")
    _locations[name] = path
    log.debug('registered path to %r handler: %r', name, path)