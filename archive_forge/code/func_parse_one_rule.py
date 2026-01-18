from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def parse_one_rule(input, skip_comments=False):
    """Parse a single :diagram:`qualified rule` or :diagram:`at-rule`.

    This would be used e.g. by `insertRule()
    <https://drafts.csswg.org/cssom/#dom-cssstylesheet-insertrule>`_
    in an implementation of CSSOM.

    :type input: :obj:`str` or :term:`iterable`
    :param input: A string or an iterable of :term:`component values`.
    :type skip_comments: :obj:`bool`
    :param skip_comments:
        If the input is a string, ignore all CSS comments.
    :returns:
        A :class:`~tinycss2.ast.QualifiedRule`,
        :class:`~tinycss2.ast.AtRule`,
        or :class:`~tinycss2.ast.ParseError` objects.

    Any whitespace or comment before or after the rule is dropped.

    """
    tokens = _to_token_iterator(input, skip_comments)
    first = _next_significant(tokens)
    if first is None:
        return ParseError(1, 1, 'empty', 'Input is empty')
    rule = _consume_rule(first, tokens)
    next = _next_significant(tokens)
    if next is not None:
        return ParseError(next.source_line, next.source_column, 'extra-input', 'Expected a single rule, got %s after the first rule.' % next.type)
    return rule