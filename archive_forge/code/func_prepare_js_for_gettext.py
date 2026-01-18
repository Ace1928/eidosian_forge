import re
def prepare_js_for_gettext(js):
    """
    Convert the JavaScript source `js` into something resembling C for
    xgettext.

    What actually happens is that all the regex literals are replaced with
    "REGEX".
    """

    def escape_quotes(m):
        """Used in a regex to properly escape double quotes."""
        s = m[0]
        if s == '"':
            return '\\"'
        else:
            return s
    lexer = JsLexer()
    c = []
    for name, tok in lexer.lex(js):
        if name == 'regex':
            tok = '"REGEX"'
        elif name == 'string':
            if tok.startswith("'"):
                guts = re.sub('\\\\.|.', escape_quotes, tok[1:-1])
                tok = '"' + guts + '"'
        elif name == 'id':
            tok = tok.replace('\\', 'U')
        c.append(tok)
    return ''.join(c)