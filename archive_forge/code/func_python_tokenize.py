import tokenize
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
def python_tokenize(code):
    code = code.replace('\n', ' ').strip()
    it = tokenize.generate_tokens(StringIO(code).readline)
    try:
        for pytype, string, (_, start), (_, end), code in it:
            if pytype == tokenize.ENDMARKER:
                break
            if pytype in (tokenize.NL, tokenize.NEWLINE):
                assert string == ''
                continue
            origin = Origin(code, start, end)
            if pytype == tokenize.ERRORTOKEN:
                raise PatsyError('error tokenizing input (maybe an unclosed string?)', origin)
            if pytype == tokenize.COMMENT:
                raise PatsyError('comments are not allowed', origin)
            yield (pytype, string, origin)
        else:
            raise ValueError('stream ended without ENDMARKER?!?')
    except tokenize.TokenError as e:
        if 'unterminated string literal' in e.args[0]:
            raise PatsyError('error tokenizing input ({})'.format(e.args[0]), Origin(code, 0, len(code)))
        assert 'EOF in multi-line' in e.args[0]
        return