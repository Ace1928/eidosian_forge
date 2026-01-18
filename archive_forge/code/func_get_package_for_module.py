from __future__ import with_statement
import datetime
import functools
import inspect
import logging
import os
import re
import sys
import six
@positional(1)
def get_package_for_module(module):
    """Get package name for a module.

    Helper calculates the package name of a module.

    Args:
      module: Module to get name for.  If module is a string, try to find
        module in sys.modules.

    Returns:
      If module contains 'package' attribute, uses that as package name.
      Else, if module is not the '__main__' module, the module __name__.
      Else, the base name of the module file name.  Else None.
    """
    if isinstance(module, six.string_types):
        try:
            module = sys.modules[module]
        except KeyError:
            return None
    try:
        return six.text_type(module.package)
    except AttributeError:
        if module.__name__ == '__main__':
            try:
                file_name = module.__file__
            except AttributeError:
                pass
            else:
                base_name = os.path.basename(file_name)
                split_name = os.path.splitext(base_name)
                if len(split_name) == 1:
                    return six.text_type(base_name)
                return u'.'.join(split_name[:-1])
        return six.text_type(module.__name__)