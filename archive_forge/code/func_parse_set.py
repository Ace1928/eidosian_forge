import collections
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_set(ctx, tokens, breakstack):
    """
  ::

    set(<variable> <value>
        [[CACHE <type> <docstring> [FORCE]] | PARENT_SCOPE])

  :see: https://cmake.org/cmake/help/v3.0/command/set.html?
  """
    tree = SetFnNode()
    while tokens and tokens[0].type in WHITESPACE_TOKENS:
        tree.children.append(tokens.pop(0))
        continue
    kwargs = {'CACHE': PositionalParser('2+', flags=['FORCE'])}
    flags = ['PARENT_SCOPE']
    kwarg_breakstack = breakstack + [KwargBreaker(list(kwargs.keys()) + flags)]
    positional_breakstack = breakstack + [KwargBreaker(list(kwargs.keys()))]
    ntokens = len(tokens)
    subtree = PositionalGroupNode.parse(ctx, tokens, 1, flags, positional_breakstack)
    assert len(tokens) < ntokens
    tree.children.append(subtree)
    for token in subtree.get_semantic_tokens():
        tree.varname = token
        break
    while tokens:
        if should_break(tokens[0], breakstack):
            break
        if tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
            if not get_tag(tokens[0]) in ('sort', 'sortable'):
                child = TreeNode(NodeType.COMMENT)
                tree.children.append(child)
                child.children.append(tokens.pop(0))
                continue
        ntokens = len(tokens)
        word = get_normalized_kwarg(tokens[0])
        if word == 'CACHE':
            with ctx.pusharg(tree):
                subtree = KeywordGroupNode.parse(ctx, tokens, word, kwargs[word], kwarg_breakstack)
            cache_tokens = subtree.get_semantic_tokens()
            cache_tokens.pop(0)
            cache_args = [None, None, False]
            for idx, token in enumerate(cache_tokens[:2]):
                cache_args[idx] = token.spelling
            if not cache_tokens:
                ctx.lint_ctx.record_lint('E1120', '<variable>', location=tokens[0].get_location())
            elif len(cache_tokens) < 2:
                ctx.lint_ctx.record_lint('E1120', '<value>', location=tokens[0].get_location())
            elif len(cache_tokens) < 3:
                pass
            elif len(cache_tokens) > 4:
                ctx.lint_ctx.record_lint('E1121', location=cache_tokens[2].get_location())
            elif cache_tokens[2].spelling.upper() == 'FORCE':
                cache_args[2] = True
            else:
                ctx.lint_ctx.record_lint('E1121', location=cache_tokens[2].get_location())
            tree.cache = CacheTuple(*cache_args)
        elif word == 'PARENT_SCOPE':
            with ctx.pusharg(tree):
                subtree = PositionalGroupNode.parse(ctx, tokens, '+', ['PARENT_SCOPE'], positional_breakstack)
            tree.parent_scope = True
        else:
            with ctx.pusharg(tree):
                subtree = PositionalGroupNode.parse(ctx, tokens, '+', [], kwarg_breakstack)
            for pattern, tags in ctx.config.parse.vartags_:
                if pattern.match(tree.varname.spelling):
                    subtree.tags.extend(tags)
            tree.value_group = subtree
        assert len(tokens) < ntokens
        tree.children.append(subtree)
    return tree