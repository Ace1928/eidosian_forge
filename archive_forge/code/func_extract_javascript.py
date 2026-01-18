from __future__ import annotations
import ast
import io
import os
import sys
import tokenize
from collections.abc import (
from os.path import relpath
from textwrap import dedent
from tokenize import COMMENT, NAME, OP, STRING, generate_tokens
from typing import TYPE_CHECKING, Any
from babel.util import parse_encoding, parse_future_flags, pathmatch
def extract_javascript(fileobj: _FileObj, keywords: Mapping[str, _Keyword], comment_tags: Collection[str], options: _JSOptions, lineno: int=1) -> Generator[_ExtractionResult, None, None]:
    """Extract messages from JavaScript source code.

    :param fileobj: the seekable, file-like object the messages should be
                    extracted from
    :param keywords: a list of keywords (i.e. function names) that should be
                     recognized as translation functions
    :param comment_tags: a list of translator tags to search for and include
                         in the results
    :param options: a dictionary of additional options (optional)
                    Supported options are:
                    * `jsx` -- set to false to disable JSX/E4X support.
                    * `template_string` -- if `True`, supports gettext(`key`)
                    * `parse_template_string` -- if `True` will parse the
                                                 contents of javascript
                                                 template strings.
    :param lineno: line number offset (for parsing embedded fragments)
    """
    from babel.messages.jslexer import Token, tokenize, unquote_string
    funcname = message_lineno = None
    messages = []
    last_argument = None
    translator_comments = []
    concatenate_next = False
    encoding = options.get('encoding', 'utf-8')
    last_token = None
    call_stack = -1
    dotted = any(('.' in kw for kw in keywords))
    for token in tokenize(fileobj.read().decode(encoding), jsx=options.get('jsx', True), template_string=options.get('template_string', True), dotted=dotted, lineno=lineno):
        if funcname and (last_token and last_token.type == 'name') and (token.type == 'template_string'):
            message_lineno = token.lineno
            messages = [unquote_string(token.value)]
            call_stack = 0
            token = Token('operator', ')', token.lineno)
        if options.get('parse_template_string') and (not funcname) and (token.type == 'template_string'):
            yield from parse_template_string(token.value, keywords, comment_tags, options, token.lineno)
        elif token.type == 'operator' and token.value == '(':
            if funcname:
                message_lineno = token.lineno
                call_stack += 1
        elif call_stack == -1 and token.type == 'linecomment':
            value = token.value[2:].strip()
            if translator_comments and translator_comments[-1][0] == token.lineno - 1:
                translator_comments.append((token.lineno, value))
                continue
            for comment_tag in comment_tags:
                if value.startswith(comment_tag):
                    translator_comments.append((token.lineno, value.strip()))
                    break
        elif token.type == 'multilinecomment':
            translator_comments = []
            value = token.value[2:-2].strip()
            for comment_tag in comment_tags:
                if value.startswith(comment_tag):
                    lines = value.splitlines()
                    if lines:
                        lines[0] = lines[0].strip()
                        lines[1:] = dedent('\n'.join(lines[1:])).splitlines()
                        for offset, line in enumerate(lines):
                            translator_comments.append((token.lineno + offset, line))
                    break
        elif funcname and call_stack == 0:
            if token.type == 'operator' and token.value == ')':
                if last_argument is not None:
                    messages.append(last_argument)
                if len(messages) > 1:
                    messages = tuple(messages)
                elif messages:
                    messages = messages[0]
                else:
                    messages = None
                if translator_comments and translator_comments[-1][0] < message_lineno - 1:
                    translator_comments = []
                if messages is not None:
                    yield (message_lineno, funcname, messages, [comment[1] for comment in translator_comments])
                funcname = message_lineno = last_argument = None
                concatenate_next = False
                translator_comments = []
                messages = []
                call_stack = -1
            elif token.type in ('string', 'template_string'):
                new_value = unquote_string(token.value)
                if concatenate_next:
                    last_argument = (last_argument or '') + new_value
                    concatenate_next = False
                else:
                    last_argument = new_value
            elif token.type == 'operator':
                if token.value == ',':
                    if last_argument is not None:
                        messages.append(last_argument)
                        last_argument = None
                    else:
                        messages.append(None)
                    concatenate_next = False
                elif token.value == '+':
                    concatenate_next = True
        elif call_stack > 0 and token.type == 'operator' and (token.value == ')'):
            call_stack -= 1
        elif funcname and call_stack == -1:
            funcname = None
        elif call_stack == -1 and token.type == 'name' and (token.value in keywords) and (last_token is None or last_token.type != 'name' or last_token.value != 'function'):
            funcname = token.value
        last_token = token