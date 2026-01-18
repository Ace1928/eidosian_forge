import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_custom_target(ctx, tokens, breakstack):
    """
  ::
    add_custom_target(Name [ALL] [command1 [args1...]]
                      [COMMAND command2 [args2...] ...]
                      [DEPENDS depend depend depend ... ]
                      [BYPRODUCTS [files...]]
                      [WORKING_DIRECTORY dir]
                      [COMMENT comment]
                      [JOB_POOL job_pool]
                      [VERBATIM] [USES_TERMINAL]
                      [COMMAND_EXPAND_LISTS]
                      [SOURCES src1 [src2...]])

  :see: https://cmake.org/cmake/help/latest/command/add_custom_target.html
  """
    kwargs = {'BYPRODUCTS': PositionalParser('+'), 'COMMAND': ShellCommandNode.parse, 'COMMENT': PositionalParser(1), 'DEPENDS': PositionalParser('+'), 'JOB_POOL': PositionalParser(1), 'SOURCES': PositionalParser('+'), 'WORKING_DIRECTORY': PositionalParser(1)}
    required_kwargs = {'COMMENT': 'C0113'}
    flags = ('VERBATIM', 'USES_TERMINAL', 'COMMAND_EXPAND_LISTS')
    tree = ArgGroupNode()
    while tokens and tokens[0].type in WHITESPACE_TOKENS:
        tree.children.append(tokens.pop(0))
        continue
    breaker = KwargBreaker(list(kwargs.keys()) + list(flags))
    child_breakstack = breakstack + [breaker]
    nametree = None
    state = 'name'
    while tokens:
        if should_break(tokens[0], breakstack):
            break
        if tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (TokenType.COMMENT, TokenType.BRACKET_COMMENT):
            child = TreeNode(NodeType.COMMENT)
            tree.children.append(child)
            child.children.append(tokens.pop(0))
            continue
        ntokens = len(tokens)
        if state == 'name':
            next_semantic = get_first_semantic_token(tokens[1:])
            if next_semantic is not None and get_normalized_kwarg(next_semantic) == 'ALL':
                npargs = 2
            else:
                npargs = 1
            nametree = PositionalGroupNode.parse(ctx, tokens, npargs, ['ALL'], child_breakstack)
            assert len(tokens) < ntokens
            tree.children.append(nametree)
            state = 'first-command'
            continue
        word = get_normalized_kwarg(tokens[0])
        if state == 'first-command':
            if not (word in kwargs or word in flags):
                subtree = PositionalGroupNode.parse(ctx, tokens, '+', [], child_breakstack)
                tree.children.append(subtree)
                assert len(tokens) < ntokens
            state = 'kwargs'
            continue
        if word in flags:
            subtree = FlagGroupNode.parse(ctx, tokens, flags, breakstack + [KwargBreaker(list(kwargs.keys()))])
            assert len(tokens) < ntokens
            tree.children.append(subtree)
            continue
        if word in kwargs:
            required_kwargs.pop(word, None)
            subtree = KeywordGroupNode.parse(ctx, tokens, word, kwargs[word], child_breakstack)
            assert len(tokens) < ntokens
            tree.children.append(subtree)
            continue
        ctx.lint_ctx.record_lint('E1122', location=tokens[0].get_location())
        subtree = PositionalGroupNode.parse(ctx, tokens, '+', [], child_breakstack)
        assert len(tokens) < ntokens
        tree.children.append(subtree)
        continue
    if required_kwargs:
        location = ()
        for token in tree.get_semantic_tokens():
            location = token.get_location()
            break
        missing_kwargs = sorted(((lintid, word) for word, lintid in sorted(required_kwargs.items())))
        for lintid, word in missing_kwargs:
            ctx.lint_ctx.record_lint(lintid, word, location=location)
    return tree