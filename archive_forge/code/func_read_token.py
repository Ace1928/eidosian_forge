import json
from ..error import GraphQLSyntaxError
def read_token(source, from_position):
    """Gets the next token from the source starting at the given position.

    This skips over whitespace and comments until it finds the next lexable
    token, then lexes punctuators immediately or calls the appropriate
    helper fucntion for more complicated tokens."""
    body = source.body
    body_length = len(body)
    position = position_after_whitespace(body, from_position)
    if position >= body_length:
        return Token(TokenKind.EOF, position, position)
    code = char_code_at(body, position)
    if code < 32 and code not in (9, 10, 13):
        raise GraphQLSyntaxError(source, position, u'Invalid character {}.'.format(print_char_code(code)))
    kind = PUNCT_CODE_TO_KIND.get(code)
    if kind is not None:
        return Token(kind, position, position + 1)
    if code == 46:
        if char_code_at(body, position + 1) == char_code_at(body, position + 2) == 46:
            return Token(TokenKind.SPREAD, position, position + 3)
    elif 65 <= code <= 90 or code == 95 or 97 <= code <= 122:
        return read_name(source, position)
    elif code == 45 or 48 <= code <= 57:
        return read_number(source, position, code)
    elif code == 34:
        return read_string(source, position)
    raise GraphQLSyntaxError(source, position, u'Unexpected character {}.'.format(print_char_code(code)))