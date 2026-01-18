from __future__ import annotations
from textwrap import dedent
from . import Extension
from ..preprocessors import Preprocessor
from .codehilite import CodeHilite, CodeHiliteExtension, parse_hl_lines
from .attr_list import get_attrs_and_remainder, AttrListExtension
from ..util import parseBoolValue
from ..serializers import _escape_attrib_html
import re
from typing import TYPE_CHECKING, Any, Iterable
def handle_attrs(self, attrs: Iterable[tuple[str, str]]) -> tuple[str, list[str], dict[str, Any]]:
    """ Return tuple: `(id, [list, of, classes], {configs})` """
    id = ''
    classes = []
    configs = {}
    for k, v in attrs:
        if k == 'id':
            id = v
        elif k == '.':
            classes.append(v)
        elif k == 'hl_lines':
            configs[k] = parse_hl_lines(v)
        elif k in self.bool_options:
            configs[k] = parseBoolValue(v, fail_on_errors=False, preserve_none=True)
        else:
            configs[k] = v
    return (id, classes, configs)