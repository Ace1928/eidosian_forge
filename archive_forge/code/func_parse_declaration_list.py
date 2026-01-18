from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def parse_declaration_list(input, skip_comments=False, skip_whitespace=False):
    """Parse a :diagram:`declaration list` (which may also contain at-rules).

    This is used e.g. for the :attr:`~tinycss2.ast.QualifiedRule.content`
    of a style rule or ``@page`` rule,
    or for the ``style`` attribute of an HTML element.

    In contexts that don’t expect any at-rule,
    all :class:`~tinycss2.ast.AtRule` objects
    should simply be rejected as invalid.

    :type input: :obj:`str` or :term:`iterable`
    :param input: A string or an iterable of :term:`component values`.
    :type skip_comments: :obj:`bool`
    :param skip_comments:
        Ignore CSS comments at the top-level of the list.
        If the input is a string, ignore all comments.
    :type skip_whitespace: :obj:`bool`
    :param skip_whitespace:
        Ignore whitespace at the top-level of the list.
        Whitespace is still preserved
        in the :attr:`~tinycss2.ast.Declaration.value` of declarations
        and the :attr:`~tinycss2.ast.AtRule.prelude`
        and :attr:`~tinycss2.ast.AtRule.content` of at-rules.
    :returns:
        A list of
        :class:`~tinycss2.ast.Declaration`,
        :class:`~tinycss2.ast.AtRule`,
        :class:`~tinycss2.ast.Comment` (if ``skip_comments`` is false),
        :class:`~tinycss2.ast.WhitespaceToken`
        (if ``skip_whitespace`` is false),
        and :class:`~tinycss2.ast.ParseError` objects

    """
    tokens = _to_token_iterator(input, skip_comments)
    result = []
    for token in tokens:
        if token.type == 'whitespace':
            if not skip_whitespace:
                result.append(token)
        elif token.type == 'comment':
            if not skip_comments:
                result.append(token)
        elif token.type == 'at-keyword':
            result.append(_consume_at_rule(token, tokens))
        elif token != ';':
            result.append(_consume_declaration_in_list(token, tokens))
    return result