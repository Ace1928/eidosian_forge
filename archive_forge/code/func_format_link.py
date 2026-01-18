import sys
from docutils import nodes
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
def format_link(name, url):
    if url:
        return f'`{name} <{url}>`_'
    return name