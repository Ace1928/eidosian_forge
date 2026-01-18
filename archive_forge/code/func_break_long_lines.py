import re
def break_long_lines(text, line_length=76):
    """
    Break lines in ASCII text text longer than line_length by inserting
    newline characters preceded by backslash (thus the resulting lines might
    be one character longer than line_length).

    The reverse operation is join_long_lines, so
    join_long_lines(break_long_lines(text)) is always returning text as long as
    consisted of ASCII characters (even when text already had lines
    ending in a backslash as those backslashes are specially escaped)!

    This is consistent with the interpretation of the backslash newline
    sequence by many languages such as magma, python, C, C++ which treat these
    escaped newlines as non-existing. In particular, the result of
    break_long_lines and the text itself is the same to these languages (as
    long as the text didn't already contain lines ending in a backslash).

    >>> text = "This is a long line.\\nThis is an even longer line.\\n"
    >>> print(break_long_lines(text,8))
    This is \\
    a long l\\
    ine.
    This is \\
    an even \\
    longer l\\
    ine.
    <BLANKLINE>

    >>> join_long_lines(break_long_lines(text, 8)) == text
    True
    """

    def split_ending_backslash(line):
        if len(line) > 0 and line[-1] == '\\':
            return (line[:-1], '\\\\\n')
        return (line, '')

    def process_line(line):
        line_without, ending_backslash = split_ending_backslash(line)
        return '\\\n'.join(_break_line_iterator(line_without, line_length)) + ending_backslash
    return '\n'.join((process_line(line) for line in text.split('\n')))