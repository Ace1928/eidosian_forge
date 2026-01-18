import logging
from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (
def parse_add_executable_standard(ctx, tokens, breakstack, sortable):
    """
  ::

    add_executable(<name> [WIN32] [MACOSX_BUNDLE]
                   [EXCLUDE_FROM_ALL]
                   [source1] [source2 ...])

  :see: https://cmake.org/cmake/help/latest/command/add_executable.html#command:add_executable
  """
    parsing_name = 1
    parsing_flags = 2
    parsing_sources = 3
    tree = ArgGroupNode()
    while tokens and tokens[0].type in WHITESPACE_TOKENS:
        tree.children.append(tokens.pop(0))
        continue
    state_ = parsing_name
    parg_group = None
    src_group = None
    active_depth = tree
    while tokens:
        if tokens[0].type is lex.TokenType.RIGHT_PAREN:
            break
        if tokens[0].type in WHITESPACE_TOKENS:
            active_depth.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
            if state_ > parsing_name:
                if get_tag(tokens[0]) in ('unsort', 'unsortable'):
                    sortable = False
                elif get_tag(tokens[0]) in ('unsort', 'unsortable'):
                    sortable = True
            child = TreeNode(NodeType.COMMENT)
            active_depth.children.append(child)
            child.children.append(tokens.pop(0))
            continue
        if state_ is parsing_name:
            token = tokens.pop(0)
            parg_group = PositionalGroupNode()
            parg_group.spec = PositionalSpec('+')
            active_depth = parg_group
            tree.children.append(parg_group)
            child = TreeNode(NodeType.ARGUMENT)
            child.children.append(token)
            CommentNode.consume_trailing(ctx, tokens, child)
            parg_group.children.append(child)
            state_ += 1
        elif state_ is parsing_flags:
            if get_normalized_kwarg(tokens[0]) in ('WIN32', 'MACOSX_BUNDLE', 'EXCLUDE_FROM_ALL'):
                token = tokens.pop(0)
                child = TreeNode(NodeType.FLAG)
                child.children.append(token)
                CommentNode.consume_trailing(ctx, tokens, child)
                parg_group.children.append(child)
            else:
                state_ += 1
                src_group = PositionalGroupNode(sortable=sortable, tags=['file-list'])
                src_group.spec = PositionalSpec('+')
                active_depth = src_group
                tree.children.append(src_group)
        elif state_ is parsing_sources:
            token = tokens.pop(0)
            child = TreeNode(NodeType.ARGUMENT)
            child.children.append(token)
            CommentNode.consume_trailing(ctx, tokens, child)
            src_group.children.append(child)
            if only_comments_and_whitespace_remain(tokens, breakstack):
                active_depth = tree
    return tree