import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def parse_component_value_list(css, skip_comments=False):
    """Parse a list of component values.

    :type css: :obj:`str`
    :param css: A CSS string.
    :type skip_comments: :obj:`bool`
    :param skip_comments:
        Ignore CSS comments.
        The return values (and recursively its blocks and functions)
        will not contain any :class:`~tinycss2.ast.Comment` object.
    :returns: A list of :term:`component values`.

    """
    css = css.replace('\x00', '�').replace('\r\n', '\n').replace('\r', '\n').replace('\x0c', '\n')
    length = len(css)
    token_start_pos = pos = 0
    line = 1
    last_newline = -1
    root = tokens = []
    end_char = None
    stack = []
    while pos < length:
        newline = css.rfind('\n', token_start_pos, pos)
        if newline != -1:
            line += 1 + css.count('\n', token_start_pos, newline)
            last_newline = newline
        column = pos - last_newline
        token_start_pos = pos
        c = css[pos]
        if c in ' \n\t':
            pos += 1
            while css.startswith((' ', '\n', '\t'), pos):
                pos += 1
            value = css[token_start_pos:pos]
            tokens.append(WhitespaceToken(line, column, value))
            continue
        elif c in 'Uu' and pos + 2 < length and (css[pos + 1] == '+') and (css[pos + 2] in '0123456789abcdefABCDEF?'):
            start, end, pos = _consume_unicode_range(css, pos + 2)
            tokens.append(UnicodeRangeToken(line, column, start, end))
            continue
        elif css.startswith('-->', pos):
            tokens.append(LiteralToken(line, column, '-->'))
            pos += 3
            continue
        elif _is_ident_start(css, pos):
            value, pos = _consume_ident(css, pos)
            if not css.startswith('(', pos):
                tokens.append(IdentToken(line, column, value))
                continue
            pos += 1
            if ascii_lower(value) == 'url':
                url_pos = pos
                while css.startswith((' ', '\n', '\t'), url_pos):
                    url_pos += 1
                if url_pos >= length or css[url_pos] not in ('"', "'"):
                    value, pos, error = _consume_url(css, pos)
                    if value is not None:
                        repr = 'url({})'.format(serialize_url(value))
                        if error is not None:
                            error_key = error[0]
                            if error_key == 'eof-in-string':
                                repr = repr[:-2]
                            else:
                                assert error_key == 'eof-in-url'
                                repr = repr[:-1]
                        tokens.append(URLToken(line, column, value, repr))
                    if error is not None:
                        tokens.append(ParseError(line, column, *error))
                    continue
            arguments = []
            tokens.append(FunctionBlock(line, column, value, arguments))
            stack.append((tokens, end_char))
            end_char = ')'
            tokens = arguments
            continue
        match = _NUMBER_RE.match(css, pos)
        if match:
            pos = match.end()
            repr_ = css[token_start_pos:pos]
            value = float(repr_)
            int_value = int(repr_) if not any(match.groups()) else None
            if pos < length and _is_ident_start(css, pos):
                unit, pos = _consume_ident(css, pos)
                tokens.append(DimensionToken(line, column, value, int_value, repr_, unit))
            elif css.startswith('%', pos):
                pos += 1
                tokens.append(PercentageToken(line, column, value, int_value, repr_))
            else:
                tokens.append(NumberToken(line, column, value, int_value, repr_))
        elif c == '@':
            pos += 1
            if pos < length and _is_ident_start(css, pos):
                value, pos = _consume_ident(css, pos)
                tokens.append(AtKeywordToken(line, column, value))
            else:
                tokens.append(LiteralToken(line, column, '@'))
        elif c == '#':
            pos += 1
            if pos < length and (css[pos] in '0123456789abcdefghijklmnopqrstuvwxyz-_ABCDEFGHIJKLMNOPQRSTUVWXYZ' or ord(css[pos]) > 127 or (css[pos] == '\\' and (not css.startswith('\\\n', pos)))):
                is_identifier = _is_ident_start(css, pos)
                value, pos = _consume_ident(css, pos)
                tokens.append(HashToken(line, column, value, is_identifier))
            else:
                tokens.append(LiteralToken(line, column, '#'))
        elif c == '{':
            content = []
            tokens.append(CurlyBracketsBlock(line, column, content))
            stack.append((tokens, end_char))
            end_char = '}'
            tokens = content
            pos += 1
        elif c == '[':
            content = []
            tokens.append(SquareBracketsBlock(line, column, content))
            stack.append((tokens, end_char))
            end_char = ']'
            tokens = content
            pos += 1
        elif c == '(':
            content = []
            tokens.append(ParenthesesBlock(line, column, content))
            stack.append((tokens, end_char))
            end_char = ')'
            tokens = content
            pos += 1
        elif c == end_char:
            tokens, end_char = stack.pop()
            pos += 1
        elif c in '}])':
            tokens.append(ParseError(line, column, c, 'Unmatched ' + c))
            pos += 1
        elif c in ('"', "'"):
            value, pos, error = _consume_quoted_string(css, pos)
            if value is not None:
                repr = '"{}"'.format(serialize_string_value(value))
                if error is not None:
                    repr = repr[:-1]
                tokens.append(StringToken(line, column, value, repr))
            if error is not None:
                tokens.append(ParseError(line, column, *error))
        elif css.startswith('/*', pos):
            pos = css.find('*/', pos + 2)
            if pos == -1:
                if not skip_comments:
                    tokens.append(Comment(line, column, css[token_start_pos + 2:]))
                break
            if not skip_comments:
                tokens.append(Comment(line, column, css[token_start_pos + 2:pos]))
            pos += 2
        elif css.startswith('<!--', pos):
            tokens.append(LiteralToken(line, column, '<!--'))
            pos += 4
        elif css.startswith('||', pos):
            tokens.append(LiteralToken(line, column, '||'))
            pos += 2
        elif c in '~|^$*':
            pos += 1
            if css.startswith('=', pos):
                pos += 1
                tokens.append(LiteralToken(line, column, c + '='))
            else:
                tokens.append(LiteralToken(line, column, c))
        else:
            tokens.append(LiteralToken(line, column, c))
            pos += 1
    return root