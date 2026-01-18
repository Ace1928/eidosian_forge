from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def asplit(stream):
    return FilterNode(stream, asplit.__name__)