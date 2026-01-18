from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
@classmethod
def parse2(cls, ctx, tokens, spec, breakstack):
    """
    Parse a continuous sequence of `npargs` positional arguments. If npargs is
    an integer we will consume exactly that many arguments. If it is not an
    integer then it is a string meaning:

    * "?": zero or one
    * "*": zero or more
    * "+": one or more
    """
    tree = cls(sortable=spec.sortable, tags=spec.tags)
    tree.spec = spec
    nconsumed = 0
    while tokens and tokens[0].type in WHITESPACE_TOKENS:
        tree.children.append(tokens.pop(0))
    if tokens and get_tag(tokens[0]) in ('sortable', 'sort'):
        tree.sortable = True
    elif tokens and get_tag(tokens[0]) in ('unsortable', 'unsort'):
        tree.sortable = False
    while tokens:
        if pargs_are_full(spec.nargs, nconsumed):
            break
        if should_break(tokens[0], breakstack):
            if not npargs_is_exact(spec.nargs):
                break
            if tokens[0].type == lex.TokenType.RIGHT_PAREN:
                break
        if tokens[0].type == lex.TokenType.LEFT_PAREN:
            with ctx.pusharg(tree):
                subtree = ParenGroupNode.parse(ctx, tokens, breakstack)
            tree.children.append(subtree)
            continue
        if tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
            if comment_belongs_up_tree(ctx, tokens, tree, breakstack):
                break
            child = CommentNode.consume(ctx, tokens)
            tree.children.append(child)
            continue
        if tokens[0].type in (lex.TokenType.FORMAT_OFF, lex.TokenType.FORMAT_ON):
            tree.children.append(OnOffNode.consume(ctx, tokens))
            continue
        if get_normalized_kwarg(tokens[0]) in spec.flags:
            child = TreeNode(NodeType.FLAG)
        else:
            child = TreeNode(NodeType.ARGUMENT)
        child.children.append(tokens.pop(0))
        CommentNode.consume_trailing(ctx, tokens, child)
        tree.children.append(child)
        nconsumed += 1
    return tree