import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
def get_all_filters():
    """Return a generator of all filter names."""
    for name in FILTERS:
        yield name
    for name, _ in find_plugin_filters():
        yield name