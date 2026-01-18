import re
def join_long_lines_deleting_whitespace(text):
    """
    Similar to join_long_lines, but also deletes whitespace following a
    backslash newline sequence.

    Programs such as magma break long integers by introducing a backslash
    newline sequence and inserting extra whitespace for formatting.
    join_long_lines_deleting_whitespace can be used to undo this and parse
    the input normally.

    >>> join_long_lines_deleting_whitespace("Text:\\\\\\n   More")
    'Text:More'
    >>> join_long_lines_deleting_whitespace("    1234\\\\\\n    5678").strip()
    '12345678'
    """
    return re.sub('\\\\\\n\\s*', '', text)