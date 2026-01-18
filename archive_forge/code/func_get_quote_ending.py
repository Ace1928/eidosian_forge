import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def get_quote_ending(string, code_lines, position, invert_result=False):
    _, quote = _get_string_prefix_and_quote(string)
    if quote is None:
        return ''
    if _matches_quote_at_position(code_lines, quote, position) != invert_result:
        return ''
    return quote