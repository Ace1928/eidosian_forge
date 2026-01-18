import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.formatters._mapping import FORMATTERS
from pygments.plugin import find_plugin_formatters
from pygments.util import ClassNotFound, itervalues
def get_formatter_for_filename(fn, **options):
    """Lookup and instantiate a formatter by filename pattern.

    Raises ClassNotFound if not found.
    """
    fn = basename(fn)
    for modname, name, _, filenames, _ in itervalues(FORMATTERS):
        for filename in filenames:
            if _fn_matches(fn, filename):
                if name not in _formatter_cache:
                    _load_formatters(modname)
                return _formatter_cache[name](**options)
    for cls in find_plugin_formatters():
        for filename in cls.filenames:
            if _fn_matches(fn, filename):
                return cls(**options)
    raise ClassNotFound('no formatter found for file name %r' % fn)