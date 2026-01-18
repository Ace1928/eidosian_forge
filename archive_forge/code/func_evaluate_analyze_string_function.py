import decimal
import os
import re
import codecs
import math
from copy import copy
from itertools import zip_longest
from typing import cast, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen
from urllib.error import URLError
from ..exceptions import ElementPathError
from ..tdop import MultiLabel
from ..helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, \
from ..namespaces import get_expanded_name, split_expanded_name, \
from ..datatypes import xsd10_atomic_types, NumericProxy, QName, Date10, \
from ..sequence_types import is_sequence_type, match_sequence_type
from ..etree import defuse_xml, etree_iter_paths
from ..xpath_nodes import XPathNode, ElementNode, TextNode, AttributeNode, \
from ..tree_builders import get_node_tree
from ..xpath_tokens import XPathFunctionArgType, XPathToken, ValueToken, XPathFunction
from ..serialization import get_serialization_params, serialize_to_xml, serialize_to_json
from ..xpath_context import XPathContext, XPathSchemaContext
from ..regex import translate_pattern, RegexError
from ._xpath30_operators import XPath30Parser
from .xpath30_helpers import UNICODE_DIGIT_PATTERN, DECIMAL_DIGIT_PATTERN, \
@method(function('analyze-string', nargs=(2, 3), sequence_types=('xs:string?', 'xs:string', 'xs:string', 'element(fn:analyze-string-result)')))
def evaluate_analyze_string_function(self, context=None):
    if self.context is not None:
        context = self.context
    input_string = self.get_argument(context, default='', cls=str)
    pattern = self.get_argument(context, 1, required=True, cls=str)
    flags = 0
    if len(self) > 2:
        for c in self.get_argument(context, 2, required=True, cls=str):
            if c in 'smix':
                flags |= getattr(re, c.upper())
            elif c == 'q' and self.parser.version > '2':
                pattern = re.escape(pattern)
            else:
                raise self.error('FORX0001', 'Invalid regular expression flag %r' % c)
    try:
        python_pattern = translate_pattern(pattern, flags, self.parser.xsd_version)
        compiled_pattern = re.compile(python_pattern, flags=flags)
    except (re.error, RegexError) as err:
        msg = 'Invalid regular expression: {}'
        raise self.error('FORX0002', msg.format(str(err))) from None
    except OverflowError as err:
        raise self.error('FORX0002', err) from None
    if compiled_pattern.match('') is not None:
        raise self.error('FORX0003', 'pattern matches a zero-length string')
    if context is None:
        raise self.missing_context()
    level = 0
    escaped = False
    char_class = False
    group_levels = [0]
    for s in compiled_pattern.pattern:
        if escaped:
            escaped = False
        elif s == '\\':
            escaped = True
        elif char_class:
            if s == ']':
                char_class = False
        elif s == '[':
            char_class = True
        elif s == '(':
            group_levels.append(level)
            level += 1
        elif s == ')':
            level -= 1
    lines = ['<analyze-string-result xmlns="{}">'.format(XPATH_FUNCTIONS_NAMESPACE)]
    k = 0
    while k < len(input_string):
        match = compiled_pattern.search(input_string, k)
        if match is None:
            lines.append('<non-match>{}</non-match>'.format(input_string[k:]))
            break
        elif not match.groups():
            start, stop = match.span()
            if start > k:
                lines.append('<non-match>{}</non-match>'.format(input_string[k:start]))
            lines.append('<match>{}</match>'.format(input_string[start:stop]))
            k = stop
        else:
            start, stop = match.span()
            if start > k:
                lines.append('<non-match>{}</non-match>'.format(input_string[k:start]))
                k = start
            match_items = []
            group_tmpl = '<group nr="{}">{}'
            empty_group_tmpl = '<group nr="{}"/>'
            unclosed_groups = 0
            for idx in range(1, compiled_pattern.groups + 1):
                _start, _stop = match.span(idx)
                if _start < 0:
                    continue
                elif _start > k:
                    if unclosed_groups:
                        for _ in range(unclosed_groups):
                            match_items.append('</group>')
                        unclosed_groups = 0
                    match_items.append(input_string[k:_start])
                if _start == _stop:
                    if group_levels[idx] <= group_levels[idx - 1]:
                        for _ in range(unclosed_groups):
                            match_items.append('</group>')
                        unclosed_groups = 0
                    match_items.append(empty_group_tmpl.format(idx))
                    k = _stop
                elif idx == compiled_pattern.groups:
                    k = _stop
                    match_items.append(group_tmpl.format(idx, input_string[_start:k]))
                    match_items.append('</group>')
                else:
                    next_start = match.span(idx + 1)[0]
                    if next_start < 0 or _stop < next_start or (_stop == next_start and group_levels[idx + 1] <= group_levels[idx]):
                        k = _stop
                        match_items.append(group_tmpl.format(idx, input_string[_start:k]))
                        match_items.append('</group>')
                    else:
                        k = next_start
                        match_items.append(group_tmpl.format(idx, input_string[_start:k]))
                        unclosed_groups += 1
            for _ in range(unclosed_groups):
                match_items.append('</group>')
            match_items.append(input_string[k:stop])
            k = stop
            lines.append('<match>{}</match>'.format(''.join(match_items)))
    lines.append('</analyze-string-result>')
    if self.parser.defuse_xml:
        root = context.etree.XML(defuse_xml(''.join(lines)))
    else:
        root = context.etree.XML(''.join(lines))
    return get_node_tree(root=root, namespaces=self.parser.namespaces)