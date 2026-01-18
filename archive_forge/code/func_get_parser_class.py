import sys
from docutils import Component
def get_parser_class(parser_name):
    """Return the Parser class from the `parser_name` module."""
    parser_name = parser_name.lower()
    if parser_name in _parser_aliases:
        parser_name = _parser_aliases[parser_name]
    try:
        module = __import__(parser_name, globals(), locals(), level=1)
    except ImportError:
        module = __import__(parser_name, globals(), locals(), level=0)
    return module.Parser