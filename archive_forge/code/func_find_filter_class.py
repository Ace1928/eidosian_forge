import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
def find_filter_class(filtername):
    """Lookup a filter by name. Return None if not found."""
    if filtername in FILTERS:
        return FILTERS[filtername]
    for name, cls in find_plugin_filters():
        if name == filtername:
            return cls
    return None