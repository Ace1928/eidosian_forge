import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def make_xml_tags(tag_str: Union[str, ParserElement]) -> Tuple[ParserElement, ParserElement]:
    """Helper to construct opening and closing tag expressions for XML,
    given a tag name. Matches tags only in the given upper/lower case.

    Example: similar to :class:`make_html_tags`
    """
    return _makeTags(tag_str, True)