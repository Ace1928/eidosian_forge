from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def parse_one_declaration(input, skip_comments=False):
    """Parse a single :diagram:`declaration`.

    This is used e.g. for a declaration in an `@supports
    <https://drafts.csswg.org/css-conditional/#at-supports>`_ test.

    :type input: :obj:`str` or :term:`iterable`
    :param input: A string or an iterable of :term:`component values`.
    :type skip_comments: :obj:`bool`
    :param skip_comments: If the input is a string, ignore all CSS comments.
    :returns:
        A :class:`~tinycss2.ast.Declaration`
        or :class:`~tinycss2.ast.ParseError`.

    Any whitespace or comment before the ``:`` colon is dropped.

    """
    tokens = _to_token_iterator(input, skip_comments)
    first_token = _next_significant(tokens)
    if first_token is None:
        return ParseError(1, 1, 'empty', 'Input is empty')
    return _parse_declaration(first_token, tokens)