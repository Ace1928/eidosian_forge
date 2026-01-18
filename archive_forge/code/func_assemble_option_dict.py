import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def assemble_option_dict(option_list, options_spec):
    """
    Return a mapping of option names to values.

    :Parameters:
        - `option_list`: A list of (name, value) pairs (the output of
          `extract_options()`).
        - `options_spec`: Dictionary mapping known option names to a
          conversion function such as `int` or `float`.

    :Exceptions:
        - `KeyError` for unknown option names.
        - `DuplicateOptionError` for duplicate options.
        - `ValueError` for invalid option values (raised by conversion
           function).
        - `TypeError` for invalid option value types (raised by conversion
           function).
    """
    options = {}
    for name, value in option_list:
        convertor = options_spec[name]
        if convertor is None:
            raise KeyError(name)
        if name in options:
            raise DuplicateOptionError('duplicate option "%s"' % name)
        try:
            options[name] = convertor(value)
        except (ValueError, TypeError) as detail:
            raise detail.__class__('(option: "%s"; value: %r)\n%s' % (name, value, ' '.join(detail.args)))
    return options