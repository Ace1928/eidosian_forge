import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
def format_field(formatter, field_name, separator_token, token_iter):
    """Format a field using a provided formatter

    This function formats a series of tokens using the provided formatter.
    It can be used as a standalone formatter engine and can be used in test
    suites to validate third-party formatters (enabling them to test for
    corner cases without involving parsing logic).

    The formatter receives series of FormatterContentTokens (via the
    token_iter) and is expected to yield one or more str or
    FormatterContentTokens.  The calling function will combine all of
    these into a single string, which will be used as the value.

    The formatter is recommended to yield the provided value and comment
    tokens interleaved with text segments of whitespace and separators
    as part of its output.  If it preserve comment and value tokens, the
    calling function can provide some runtime checks to catch bugs
    (like the formatter turning a comment into a value because it forgot
    to ensure that the comment was emitted directly after a newline
    character).

    When writing a formatter, please keep the following in mind:

     * The output of the formatter is appended directly after the ":" separator.
       Most formatters will want to emit either a space or a newline as the very
       first character for readability.
       (compare "Depends:foo\\n" to "Depends: foo\\n")

     * The formatter must always end its output on a newline.  This is a design
       choice of how the round-trip safe parser represent values that is imposed
       on the formatter.

     * It is often easier to discard/ignore all separator tokens from the
       the provided token sequence and instead just yield separator tokens/str
       where the formatter wants to place them.

         - The formatter is strongly recommended to special-case formatting
           for whitespace separators (check for `separator_token.is_whitespace`).

           This is because space, tab and newline all counts as valid separators
           and can all appear in the token sequence. If the original field uses
           a mix of these separators it is likely to completely undermine the
           desired result. Not to mention the additional complexity of handling
           when a separator token happens to use the newline character which
           affects how the formatter is supposed what comes after it
           (see the rules for comments, empty lines and continuation line
           markers).

     * The formatter must remember to emit a "continuation line" marker
       (typically a single space or tab) when emitting a value after
       a newline or a comment. A `yield " "` is sufficient.

        - The continuation line marker may be embedded inside a str
          with other whitespace (such as the newline coming before it
          or/and whitespace used for indentation purposes following
          the marker).

     * The formatter must not cause the output to contain completely
       empty/whitespace lines as these cause syntax errors.  The first
       line never counts as an empty line (as it will be appended after
       the field name).

     * Tokens must be discriminated via the `token.is_value` (etc.)
       properties. Assuming that `token.text.startswith("#")` implies a
       comment and similar stunts are wrong.  As an example, "#foo" is a
       perfectly valid value in some contexts.

     * Comment tokens *always* take up exactly one complete line including
       the newline character at the end of the line. They must be emitted
       directly after a newline character or another comment token.

     * Special cases that are rare but can happen:

       - Fields *can* start with comments and requires a formatter provided newline.
         (Example: "Depends:\\n# Comment here\\n foo")

       - Fields *can* start on a separator or have two separators in a row.
         This is especially true for whitespace separated fields where every
         whitespace counts as a separator, but it can also happen with other
         separators (such as comma).

       - Value tokens can contain whitespace (for non-whitespace separators).
         When they do, the formatter must not attempt change nor "normalize"
         the whitespace inside the value token as that might change how the
         value is interpreted.  (If you want to normalize such whitespace,
         the formatter is at the wrong abstraction level.  Instead, manipulate
         the values directly in the value interpretation layer)

    This function will provide *some* runtime checks of its input and the
    output from the formatter to detect some errors early and provide
    helpful diagnostics.  If you use the function for testing, you are
    recommended to rely on verifying the output of the function rather than
    relying on the runtime checks (as these are subject to change).

    :param formatter: A formatter (see FormatterCallback for the type).
    Basic formatting is provided via one_value_per_line_trailing_separator
    (a formatter) or one_value_per_line_formatter (a formatter generator).
    :param field_name: The name of the field.
    :param separator_token: One of SPACE_SEPARATOR and COMMA_SEPARATOR
    :param token_iter: An iterable of tokens to be formatted.

    The following example shows how to define a formatter_callback along with
    a few verifications.

    >>> fmt_field_len_sep = one_value_per_line_trailing_separator
    >>> fmt_shortest = one_value_per_line_formatter(
    ...   1,
    ...   trailing_separator=False
    ... )
    >>> fmt_newline_first = one_value_per_line_formatter(
    ...   1,
    ...   trailing_separator=False,
    ...   immediate_empty_line=True
    ... )
    >>> # Omit separator tokens for in the token list for simplicity (the formatter does
    >>> # not use them, and it enables us to keep the example simple by reusing the list)
    >>> tokens = [
    ...     FormatterContentToken.value_token("foo"),
    ...     FormatterContentToken.comment_token("# some comment about bar\\n"),
    ...     FormatterContentToken.value_token("bar"),
    ... ]
    >>> # Starting with fmt_dl_ts
    >>> print(format_field(fmt_field_len_sep, "Depends", COMMA_SEPARATOR_FT, tokens), end='')
    Depends: foo,
    # some comment about bar
             bar,
    >>> print(format_field(fmt_field_len_sep, "Architecture", SPACE_SEPARATOR_FT, tokens), end='')
    Architecture: foo
    # some comment about bar
                  bar
    >>> # Control check for the special case where the field starts with a comment
    >>> print(format_field(fmt_field_len_sep, "Depends", COMMA_SEPARATOR_FT, tokens[1:]), end='')
    Depends:
    # some comment about bar
             bar,
    >>> # Also, check single line values (to ensure it ends on a newline)
    >>> print(format_field(fmt_field_len_sep, "Depends", COMMA_SEPARATOR_FT, tokens[2:]), end='')
    Depends: bar,
    >>> ### Changing format to the shortest length
    >>> print(format_field(fmt_shortest, "Depends", COMMA_SEPARATOR_FT, tokens), end='')
    Depends: foo,
    # some comment about bar
     bar
    >>> print(format_field(fmt_shortest, "Architecture", SPACE_SEPARATOR_FT, tokens), end='')
    Architecture: foo
    # some comment about bar
     bar
    >>> # Control check for the special case where the field starts with a comment
    >>> print(format_field(fmt_shortest, "Depends", COMMA_SEPARATOR_FT, tokens[1:]), end='')
    Depends:
    # some comment about bar
     bar
    >>> # Also, check single line values (to ensure it ends on a newline)
    >>> print(format_field(fmt_shortest, "Depends", COMMA_SEPARATOR_FT, tokens[2:]), end='')
    Depends: bar
    >>> ### Changing format to the newline first format
    >>> print(format_field(fmt_newline_first, "Depends", COMMA_SEPARATOR_FT, tokens), end='')
    Depends:
     foo,
    # some comment about bar
     bar
    >>> print(format_field(fmt_newline_first, "Architecture", SPACE_SEPARATOR_FT, tokens), end='')
    Architecture:
     foo
    # some comment about bar
     bar
    >>> # Control check for the special case where the field starts with a comment
    >>> print(format_field(fmt_newline_first, "Depends", COMMA_SEPARATOR_FT, tokens[1:]), end='')
    Depends:
    # some comment about bar
     bar
    >>> # Also, check single line values (to ensure it ends on a newline)
    >>> print(format_field(fmt_newline_first, "Depends", COMMA_SEPARATOR_FT, tokens[2:]), end='')
    Depends:
     bar
    """
    formatted_tokens = [field_name, ':']
    just_after_newline = False
    last_was_value_token = False
    if isinstance(token_iter, list):
        last_token = token_iter[-1]
        if last_token.is_comment:
            raise ValueError('Invalid token_iter: Field values cannot end with comments')
    for token in formatter(field_name, separator_token, token_iter):
        token_as_text = str(token)
        if isinstance(token, FormatterContentToken):
            if token.is_comment:
                if not just_after_newline:
                    raise ValueError('Bad format: Comments must appear directly after a newline.')
                if not token_as_text.startswith('#'):
                    raise ValueError('Invalid Comment token: Must start with #')
                if not token_as_text.endswith('\n'):
                    raise ValueError('Invalid Comment token: Must end on a newline')
            elif token.is_value:
                if token_as_text[0].isspace() or token_as_text[-1].isspace():
                    raise ValueError('Invalid Value token: It cannot start nor end on whitespace')
                if just_after_newline:
                    raise ValueError('Bad format: Missing continuation line marker')
                if last_was_value_token:
                    raise ValueError('Bad format: Formatter omitted a separator')
            last_was_value_token = token.is_value
        else:
            last_was_value_token = False
        if just_after_newline:
            if token_as_text[0] in ('\r', '\n'):
                raise ValueError('Bad format: Saw completely empty line.')
            if not token_as_text[0].isspace() and (not token_as_text.startswith('#')):
                raise ValueError('Bad format: Saw completely empty line.')
        formatted_tokens.append(token_as_text)
        just_after_newline = token_as_text.endswith('\n')
    formatted_text = ''.join(formatted_tokens)
    if not formatted_text.endswith('\n'):
        raise ValueError('Bad format: The field value must end on a newline')
    return formatted_text