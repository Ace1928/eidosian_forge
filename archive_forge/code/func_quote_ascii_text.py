import re
def quote_ascii_text(text):
    """
    Put the text in double quotes after escapes newlines, backslashes and
    double quotes. Giving the result of quote_ascii_text to eval should give
    the original string back if the string contained only ASCII characters.

    Similarly, giving the result of quote_ascii_text to magma's print,
    should give the original string back (magma's print might wrap long lines
    though).

    >>> text = 'Backslash:\\\\, Newline:\\n, Quote: "'
    >>> quote_ascii_text(text)
    '"Backslash:\\\\\\\\, Newline:\\\\n, Quote: \\\\""'
    >>> eval(quote_ascii_text(text)) == text
    True
    """

    def process_char(char):
        if char == '\n':
            return '\\n'
        if char == '\\':
            return '\\\\'
        if char == '"':
            return '\\"'
        return char
    return '"' + ''.join((process_char(c) for c in text)) + '"'