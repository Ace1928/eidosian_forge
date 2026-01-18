import json
from ..error import GraphQLSyntaxError
def char_code_at(s, pos):
    if 0 <= pos < len(s):
        return ord(s[pos])
    return None