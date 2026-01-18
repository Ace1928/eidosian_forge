from __future__ import annotations
import logging
import re
from ..token import Token
from .state_core import StateCore
def replace_rare(inlineTokens: list[Token]) -> None:
    inside_autolink = 0
    for token in inlineTokens:
        if token.type == 'text' and (not inside_autolink) and RARE_RE.search(token.content):
            token.content = PLUS_MINUS_RE.sub('±', token.content)
            token.content = ELLIPSIS_RE.sub('…', token.content)
            token.content = ELLIPSIS_QUESTION_EXCLAMATION_RE.sub('\\1..', token.content)
            token.content = QUESTION_EXCLAMATION_RE.sub('\\1\\1\\1', token.content)
            token.content = COMMA_RE.sub(',', token.content)
            token.content = EM_DASH_RE.sub('\\1—', token.content)
            token.content = EN_DASH_RE.sub('\\1–', token.content)
            token.content = EN_DASH_INDENT_RE.sub('\\1–', token.content)
        if token.type == 'link_open' and token.info == 'auto':
            inside_autolink -= 1
        if token.type == 'link_close' and token.info == 'auto':
            inside_autolink += 1