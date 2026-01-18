from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_cmake_policy(ctx, tokens, breakstack):
    """
  ::

    cmake_policy(VERSION <min>[...<max>])
    cmake_policy(SET CMP<NNNN> NEW)
    cmake_policy(SET CMP<NNNN> OLD)
    cmake_policy(GET CMP<NNNN> <variable>)
    cmake_policy(PUSH)
    cmake_policy(POP)

  :see: https://cmake.org/cmake/help/latest/command/cmake_policy.html
  """
    first_token = get_nth_semantic_token(tokens, 0)
    if first_token is None:
        return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)
    if first_token.type is lex.TokenType.DEREF:
        ctx.lint_ctx.record_lint('C0114', location=first_token.get_location())
        return StandardArgTree.parse(ctx, tokens, '+', {}, [], breakstack)
    descriminator = first_token.spelling.upper()
    if descriminator == 'VERSION':
        return StandardArgTree.parse(ctx, tokens, '2+', {}, ['VERSION'], breakstack)
    if descriminator == 'SET':
        return StandardArgTree.parse(ctx, tokens, 3, {}, ['SET', 'OLD', 'NEW'], breakstack)
    if descriminator == 'GET':
        return StandardArgTree.parse(ctx, tokens, 3, {}, ['GET'], breakstack)
    if descriminator in ('PUSH', 'POP'):
        return StandardArgTree.parse(ctx, tokens, 1, {}, ['PUSH', 'POP'], breakstack)
    ctx.lint_ctx.record_lint('E1126', location=first_token.get_location())
    return StandardArgTree.parse(ctx, tokens, 1, {}, ['PUSH', 'POP'], breakstack)