import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.formatters._mapping import FORMATTERS
from pygments.plugin import find_plugin_formatters
from pygments.util import ClassNotFound, itervalues
def get_formatter_by_name(_alias, **options):
    """Lookup and instantiate a formatter by alias.

    Raises ClassNotFound if not found.
    """
    cls = find_formatter_class(_alias)
    if cls is None:
        raise ClassNotFound('no formatter found for name %r' % _alias)
    return cls(**options)