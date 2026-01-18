import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.formatters._mapping import FORMATTERS
from pygments.plugin import find_plugin_formatters
from pygments.util import ClassNotFound, itervalues
def find_formatter_class(alias):
    """Lookup a formatter by alias.

    Returns None if not found.
    """
    for module_name, name, aliases, _, _ in itervalues(FORMATTERS):
        if alias in aliases:
            if name not in _formatter_cache:
                _load_formatters(module_name)
            return _formatter_cache[name]
    for _, cls in find_plugin_formatters():
        if alias in cls.aliases:
            return cls