from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def analyze_scalar(self, scalar):
    if not scalar:
        return ScalarAnalysis(scalar=scalar, empty=True, multiline=False, allow_flow_plain=False, allow_block_plain=True, allow_single_quoted=True, allow_double_quoted=True, allow_block=False)
    block_indicators = False
    flow_indicators = False
    line_breaks = False
    special_characters = False
    leading_space = False
    leading_break = False
    trailing_space = False
    trailing_break = False
    break_space = False
    space_break = False
    if scalar.startswith(u'---') or scalar.startswith(u'...'):
        block_indicators = True
        flow_indicators = True
    preceeded_by_whitespace = True
    followed_by_whitespace = len(scalar) == 1 or scalar[1] in u'\x00 \t\r\n\x85\u2028\u2029'
    previous_space = False
    previous_break = False
    index = 0
    while index < len(scalar):
        ch = scalar[index]
        if index == 0:
            if ch in u'#,[]{}&*!|>\'"%@`':
                flow_indicators = True
                block_indicators = True
            if ch in u'?:':
                if self.serializer.use_version == (1, 1):
                    flow_indicators = True
                elif len(scalar) == 1:
                    flow_indicators = True
                if followed_by_whitespace:
                    block_indicators = True
            if ch == u'-' and followed_by_whitespace:
                flow_indicators = True
                block_indicators = True
        else:
            if ch in u',[]{}':
                flow_indicators = True
            if ch == u'?' and self.serializer.use_version == (1, 1):
                flow_indicators = True
            if ch == u':':
                if followed_by_whitespace:
                    flow_indicators = True
                    block_indicators = True
            if ch == u'#' and preceeded_by_whitespace:
                flow_indicators = True
                block_indicators = True
        if ch in u'\n\x85\u2028\u2029':
            line_breaks = True
        if not (ch == u'\n' or u' ' <= ch <= u'~'):
            if (ch == u'\x85' or u'\xa0' <= ch <= u'\ud7ff' or u'\ue000' <= ch <= u'�' or (self.unicode_supplementary and u'𐀀' <= ch <= u'\U0010ffff')) and ch != u'\ufeff':
                if not self.allow_unicode:
                    special_characters = True
            else:
                special_characters = True
        if ch == u' ':
            if index == 0:
                leading_space = True
            if index == len(scalar) - 1:
                trailing_space = True
            if previous_break:
                break_space = True
            previous_space = True
            previous_break = False
        elif ch in u'\n\x85\u2028\u2029':
            if index == 0:
                leading_break = True
            if index == len(scalar) - 1:
                trailing_break = True
            if previous_space:
                space_break = True
            previous_space = False
            previous_break = True
        else:
            previous_space = False
            previous_break = False
        index += 1
        preceeded_by_whitespace = ch in u'\x00 \t\r\n\x85\u2028\u2029'
        followed_by_whitespace = index + 1 >= len(scalar) or scalar[index + 1] in u'\x00 \t\r\n\x85\u2028\u2029'
    allow_flow_plain = True
    allow_block_plain = True
    allow_single_quoted = True
    allow_double_quoted = True
    allow_block = True
    if leading_space or leading_break or trailing_space or trailing_break:
        allow_flow_plain = allow_block_plain = False
    if trailing_space:
        allow_block = False
    if break_space:
        allow_flow_plain = allow_block_plain = allow_single_quoted = False
    if special_characters:
        allow_flow_plain = allow_block_plain = allow_single_quoted = allow_block = False
    elif space_break:
        allow_flow_plain = allow_block_plain = allow_single_quoted = False
        if not self.allow_space_break:
            allow_block = False
    if line_breaks:
        allow_flow_plain = allow_block_plain = False
    if flow_indicators:
        allow_flow_plain = False
    if block_indicators:
        allow_block_plain = False
    return ScalarAnalysis(scalar=scalar, empty=False, multiline=line_breaks, allow_flow_plain=allow_flow_plain, allow_block_plain=allow_block_plain, allow_single_quoted=allow_single_quoted, allow_double_quoted=allow_double_quoted, allow_block=allow_block)