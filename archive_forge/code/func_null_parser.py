import logging
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
def null_parser(ctx, tokens, breakstack):
    return None