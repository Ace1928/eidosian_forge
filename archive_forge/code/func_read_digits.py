import json
from ..error import GraphQLSyntaxError
def read_digits(source, start, first_code):
    body = source.body
    position = start
    code = first_code
    if code is not None and 48 <= code <= 57:
        while True:
            position += 1
            code = char_code_at(body, position)
            if not (code is not None and 48 <= code <= 57):
                break
        return position
    raise GraphQLSyntaxError(source, position, u'Invalid number, expected digit but got: {}.'.format(print_char_code(code)))