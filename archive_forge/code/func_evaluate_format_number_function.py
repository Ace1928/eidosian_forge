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
@method(function('format-number', nargs=(2, 3), sequence_types=('xs:numeric?', 'xs:string', 'xs:string?', 'xs:string')))
def evaluate_format_number_function(self, context=None):
    if self.context is not None:
        context = self.context
    value = self.get_argument(context, cls=NumericProxy)
    picture = self.get_argument(context, index=1, required=True, cls=str)
    decimal_format_name = self.get_argument(context, index=2, cls=str)
    if decimal_format_name is not None:
        decimal_format_name = decimal_format_name.strip()
        if decimal_format_name.startswith('Q{'):
            if decimal_format_name.startswith('Q{}'):
                decimal_format_name = decimal_format_name[3:]
            else:
                decimal_format_name = decimal_format_name[1:]
        elif ':' in decimal_format_name:
            try:
                decimal_format_name = get_expanded_name(qname=decimal_format_name, namespaces=self.parser.namespaces)
            except (KeyError, ValueError):
                raise self.error('FODF1280') from None
    try:
        decimal_format = self.parser.decimal_formats[decimal_format_name]
    except KeyError:
        raise self.error('FODF1280') from None
    pattern_separator = decimal_format['pattern-separator']
    sub_pictures = picture.split(pattern_separator)
    if len(sub_pictures) > 2:
        raise self.error('FODF1310')
    decimal_separator = decimal_format['decimal-separator']
    if any((p.count(decimal_separator) > 1 for p in sub_pictures)):
        raise self.error('FODF1310')
    percent_sign = decimal_format['percent']
    per_mille_sign = decimal_format['per-mille']
    if any((p.count(percent_sign) + p.count(per_mille_sign) > 1 for p in sub_pictures)):
        raise self.error('FODF1310')
    zero_digit = decimal_format['zero-digit']
    optional_digit = decimal_format['digit']
    digits_family = ''.join((chr(cp + ord(zero_digit)) for cp in range(10)))
    if any((optional_digit not in p and all((x not in p for x in digits_family)) for p in sub_pictures)):
        raise self.error('FODF1310')
    grouping_separator = decimal_format['grouping-separator']
    adjacent_pattern = re.compile('[\\\\%s\\\\%s]{2}' % (grouping_separator, decimal_separator))
    if any((adjacent_pattern.search(p) for p in sub_pictures)):
        raise self.error('FODF1310')
    if any((x.endswith(grouping_separator) for s in sub_pictures for x in s.split(decimal_separator))):
        raise self.error('FODF1310')
    active_characters = digits_family + ''.join([decimal_separator, grouping_separator, pattern_separator, optional_digit])
    exponent_pattern = None
    exponent_separator = 'e'
    if self.parser.version > '3.0':
        exponent_separator = decimal_format['exponent-separator']
        _pattern = re.compile('(?<=[{0}]){1}[{0}]'.format(re.escape(active_characters), exponent_separator))
        for p in sub_pictures:
            for match in _pattern.finditer(p):
                if percent_sign in p or per_mille_sign in p:
                    raise self.error('FODF1310')
                elif any((c not in digits_family for c in p[match.span()[1] - 1:])):
                    has_suffix = False
                    for ch in p[match.span()[1] - 1:]:
                        if ch in digits_family:
                            if has_suffix:
                                raise self.error('FODF1310')
                        elif ch in active_characters:
                            raise self.error('FODF1310')
                        else:
                            has_suffix = True
                exponent_pattern = _pattern
    if exponent_pattern is None:
        if any((EXPONENT_PIC.search(s) for s in sub_pictures)):
            raise self.error('FODF1310')
    if value is None or math.isnan(value):
        return decimal_format['NaN']
    elif isinstance(value, float):
        value = decimal.Decimal.from_float(value)
    elif not isinstance(value, decimal.Decimal):
        value = decimal.Decimal(value)
    minus_sign = decimal_format['minus-sign']
    prefix = ''
    if value >= 0:
        subpic = sub_pictures[0]
    else:
        subpic = sub_pictures[-1]
        if len(sub_pictures) == 1:
            prefix = minus_sign
    for k, ch in enumerate(subpic):
        if ch in active_characters:
            prefix += subpic[:k]
            subpic = subpic[k:]
            break
    else:
        prefix += subpic
        subpic = ''
    if not subpic:
        suffix = ''
    elif subpic[-1] == percent_sign:
        suffix = percent_sign
        subpic = subpic[:-1]
        if value.as_tuple().exponent < 0:
            value *= 100
        else:
            value = decimal.Decimal(int(value) * 100)
    elif subpic[-1] == per_mille_sign:
        suffix = per_mille_sign
        subpic = subpic[:-1]
        if value.as_tuple().exponent < 0:
            value *= 1000
        else:
            value = decimal.Decimal(int(value) * 1000)
    else:
        for k, ch in enumerate(reversed(subpic)):
            if ch in active_characters:
                idx = len(subpic) - k
                suffix = subpic[idx:]
                subpic = subpic[:idx]
                break
        else:
            suffix = subpic
            subpic = ''
    exp_fmt = None
    if exponent_pattern is not None:
        exp_match = exponent_pattern.search(subpic)
        if exp_match is not None:
            exp_fmt = subpic[exp_match.span()[0] + 1:]
            subpic = subpic[:exp_match.span()[0]]
    fmt_tokens = subpic.split(decimal_separator)
    if all((not fmt for fmt in fmt_tokens)):
        raise self.error('FODF1310')
    if math.isinf(value):
        return prefix + decimal_format['infinity'] + suffix
    exp_value = 0
    if exp_fmt and value:
        num_digits = 0
        for ch in fmt_tokens[0]:
            if ch in digits_family:
                num_digits += 1
        if abs(value) > 1:
            v = abs(value)
            while v > 10 ** num_digits:
                exp_value += 1
                v /= 10
            if not num_digits:
                if len(fmt_tokens) == 1:
                    fmt_tokens.append(zero_digit)
                elif not fmt_tokens[-1]:
                    fmt_tokens[-1] = zero_digit
        elif len(fmt_tokens) > 1 and fmt_tokens[-1] and (value >= 0):
            v = abs(value) * 10
            while v < 10 ** num_digits:
                exp_value -= 1
                v *= 10
        else:
            v = abs(value) * 10
            while v < 10:
                exp_value -= 1
                v *= 10
        if exp_value:
            value = value * decimal.Decimal(10) ** (-exp_value)
    if len(fmt_tokens) == 1 or not fmt_tokens[-1]:
        exp = decimal.Decimal('1')
    else:
        k = -1
        for ch in fmt_tokens[-1]:
            if ch in digits_family or ch == optional_digit:
                k += 1
        exp = decimal.Decimal('.' + '0' * k + '1')
    try:
        if value > 0:
            value = value.quantize(exp, rounding='ROUND_HALF_UP')
        else:
            value = value.quantize(exp, rounding='ROUND_HALF_DOWN')
    except decimal.InvalidOperation:
        pass
    chunks = decimal_to_string(value).lstrip('-').split('.')
    kwargs = {'digits_family': digits_family, 'optional_digit': optional_digit, 'grouping_separator': grouping_separator}
    result = format_digits(chunks[0], fmt_tokens[0], **kwargs)
    if len(fmt_tokens) > 1 and fmt_tokens[-1]:
        has_optional_digit = False
        for ch in fmt_tokens[-1]:
            if ch == optional_digit:
                has_optional_digit = True
            elif ch.isdigit() and has_optional_digit:
                raise self.error('FODF1310')
        if len(chunks) == 1:
            chunks.append(zero_digit)
        decimal_part = format_digits(chunks[1], fmt_tokens[-1], **kwargs)
        for ch in reversed(fmt_tokens[-1]):
            if ch == optional_digit:
                if decimal_part and decimal_part[-1] == zero_digit:
                    decimal_part = decimal_part[:-1]
            else:
                if not decimal_part:
                    decimal_part = zero_digit
                break
        if decimal_part:
            result += decimal_separator + decimal_part
            if not fmt_tokens[0] and result.startswith(zero_digit):
                result = result.lstrip(zero_digit)
    if exp_fmt:
        exp_digits = format_digits(str(abs(exp_value)), exp_fmt, **kwargs)
        if exp_value >= 0:
            result += f'{exponent_separator}{exp_digits}'
        else:
            result += f'{exponent_separator}-{exp_digits}'
    return prefix + result + suffix