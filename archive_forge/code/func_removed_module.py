import functools
import inspect
import wrapt
from debtcollector import _utils
def removed_module(module, replacement=None, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Helper to be called inside a module to emit a deprecation warning

    :param str replacment: A location (or information about) of any potential
                           replacement for the removed module (if applicable)
    :param str message: A message to include in the deprecation warning
    :param str version: Specify what version the removed module is present in
    :param str removal_version: What version the module will be removed. If
                                '?' is used this implies an undefined future
                                version
    :param int stacklevel: How many entries deep in the call stack before
                           ignoring
    :param type category: warnings message category (this defaults to
                          ``DeprecationWarning`` when none is provided)
    """
    if inspect.ismodule(module):
        module_name = _get_qualified_name(module)
    elif isinstance(module, str):
        module_name = module
    else:
        _qual, type_name = _utils.get_qualified_name(type(module))
        raise TypeError("Unexpected module type '%s' (expected string or module type only)" % type_name)
    prefix = "The '%s' module usage is deprecated" % module_name
    if replacement:
        postfix = ', please use %s instead' % replacement
    else:
        postfix = None
    out_message = _utils.generate_message(prefix, postfix=postfix, message=message, version=version, removal_version=removal_version)
    _utils.deprecation(out_message, stacklevel=stacklevel, category=category)