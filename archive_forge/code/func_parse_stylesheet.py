from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def parse_stylesheet(input, skip_comments=False, skip_whitespace=False):
    """Parse :diagram:`stylesheet` from text.

    This is used e.g. for a ``<style>`` HTML element.

    This differs from :func:`parse_rule_list` in that
    top-level ``<!--`` and ``-->`` tokens are ignored.
    This is a legacy quirk for the ``<style>`` HTML element.

    :type input: :obj:`str` or :term:`iterable`
    :param input: A string or an iterable of :term:`component values`.
    :type skip_comments: :obj:`bool`
    :param skip_comments:
        Ignore CSS comments at the top-level of the stylesheet.
        If the input is a string, ignore all comments.
    :type skip_whitespace: :obj:`bool`
    :param skip_whitespace:
        Ignore whitespace at the top-level of the stylesheet.
        Whitespace is still preserved
        in the :attr:`~tinycss2.ast.QualifiedRule.prelude`
        and the :attr:`~tinycss2.ast.QualifiedRule.content` of rules.
    :returns:
        A list of
        :class:`~tinycss2.ast.QualifiedRule`,
        :class:`~tinycss2.ast.AtRule`,
        :class:`~tinycss2.ast.Comment` (if ``skip_comments`` is false),
        :class:`~tinycss2.ast.WhitespaceToken`
        (if ``skip_whitespace`` is false),
        and :class:`~tinycss2.ast.ParseError` objects.

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
        elif token not in ('<!--', '-->'):
            result.append(_consume_rule(token, tokens))
    return result