from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def comment_belongs_up_tree(ctx, tokens, node, breakstack):
    """
  Return true if the comment token at tokens[0] belongs to some parent in the
  ancestry of `node` rather than to `node` itself. We determine this if the
  following is true:

  1. There is a semantic token remaining in the token stream
  2. The next semantic token would break the current scope of the node
  3. The column of the coment token is to the left of the column where the
     node starts

  e.g.

  ~~~
  statement_name(
    KEYWORD1 argument
             argument
             # comment 1
      # comment 2
    # comment 3
  )
  ~~~

  In this example:

  * comment 1 belongs to the positional group child of the keyword group
  * comment 2 belongs to the keyword group
  * comment 3 belongs to the statement
  """
    next_semantic = get_first_semantic_token(tokens)
    if next_semantic is None:
        return False
    if not should_break(next_semantic, breakstack):
        return False
    if not ctx.argstack:
        return False
    return tokens[0].get_location().col < node.get_location().col