from __future__ import absolute_import, unicode_literals
def make_lexicon():
    from ..Plex import Str, Any, AnyBut, AnyChar, Rep, Rep1, Opt, Bol, Eol, Eof, TEXT, IGNORE, Method, State, Lexicon, Range
    nonzero_digit = Any('123456789')
    digit = Any('0123456789')
    bindigit = Any('01')
    octdigit = Any('01234567')
    hexdigit = Any('0123456789ABCDEFabcdef')
    indentation = Bol + Rep(Any(' \t'))
    unicode_start_character = Any(unicode_start_ch_any) | Range(unicode_start_ch_range)
    unicode_continuation_character = unicode_start_character | Any(unicode_continuation_ch_any) | Range(unicode_continuation_ch_range)

    def underscore_digits(d):
        return Rep1(d) + Rep(Str('_') + Rep1(d))

    def prefixed_digits(prefix, digits):
        return prefix + Opt(Str('_')) + underscore_digits(digits)
    decimal = underscore_digits(digit)
    dot = Str('.')
    exponent = Any('Ee') + Opt(Any('+-')) + decimal
    decimal_fract = decimal + dot + Opt(decimal) | dot + decimal
    name = unicode_start_character + Rep(unicode_continuation_character)
    intconst = prefixed_digits(nonzero_digit, digit) | Str('0') + (prefixed_digits(Any('Xx'), hexdigit) | prefixed_digits(Any('Oo'), octdigit) | prefixed_digits(Any('Bb'), bindigit)) | underscore_digits(Str('0')) | Rep1(digit)
    intsuffix = Opt(Any('Uu')) + Opt(Any('Ll')) + Opt(Any('Ll')) | Opt(Any('Ll')) + Opt(Any('Ll')) + Opt(Any('Uu'))
    intliteral = intconst + intsuffix
    fltconst = decimal_fract + Opt(exponent) | decimal + exponent
    imagconst = (intconst | fltconst) + Any('jJ')
    beginstring = Opt(Rep(Any(string_prefixes + raw_prefixes)) | Any(char_prefixes)) + (Str("'") | Str('"') | Str("'''") | Str('"""'))
    two_oct = octdigit + octdigit
    three_oct = octdigit + octdigit + octdigit
    two_hex = hexdigit + hexdigit
    four_hex = two_hex + two_hex
    escapeseq = Str('\\') + (two_oct | three_oct | Str('N{') + Rep(AnyBut('}')) + Str('}') | Str('u') + four_hex | Str('x') + two_hex | Str('U') + four_hex + four_hex | AnyChar)
    bra = Any('([{')
    ket = Any(')]}')
    ellipsis = Str('...')
    punct = Any(':,;+-*/|&<>=.%`~^?!@')
    diphthong = Str('==', '<>', '!=', '<=', '>=', '<<', '>>', '**', '//', '+=', '-=', '*=', '/=', '%=', '|=', '^=', '&=', '<<=', '>>=', '**=', '//=', '->', '@=', '&&', '||', ':=')
    spaces = Rep1(Any(' \t\x0c'))
    escaped_newline = Str('\\\n')
    lineterm = Eol + Opt(Str('\n'))
    comment = Str('#') + Rep(AnyBut('\n'))
    return Lexicon([(name, Method('normalize_ident')), (intliteral, Method('strip_underscores', symbol='INT')), (fltconst, Method('strip_underscores', symbol='FLOAT')), (imagconst, Method('strip_underscores', symbol='IMAG')), (ellipsis | punct | diphthong, TEXT), (bra, Method('open_bracket_action')), (ket, Method('close_bracket_action')), (lineterm, Method('newline_action')), (beginstring, Method('begin_string_action')), (comment, IGNORE), (spaces, IGNORE), (escaped_newline, IGNORE), State('INDENT', [(comment + lineterm, Method('commentline')), (Opt(spaces) + Opt(comment) + lineterm, IGNORE), (indentation, Method('indentation_action')), (Eof, Method('eof_action'))]), State('SQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('\'"\n\\')), 'CHARS'), (Str('"'), 'CHARS'), (Str('\n'), Method('unclosed_string_action')), (Str("'"), Method('end_string_action')), (Eof, 'EOF')]), State('DQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('"\n\\')), 'CHARS'), (Str("'"), 'CHARS'), (Str('\n'), Method('unclosed_string_action')), (Str('"'), Method('end_string_action')), (Eof, 'EOF')]), State('TSQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('\'"\n\\')), 'CHARS'), (Any('\'"'), 'CHARS'), (Str('\n'), 'NEWLINE'), (Str("'''"), Method('end_string_action')), (Eof, 'EOF')]), State('TDQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('"\'\n\\')), 'CHARS'), (Any('\'"'), 'CHARS'), (Str('\n'), 'NEWLINE'), (Str('"""'), Method('end_string_action')), (Eof, 'EOF')]), (Eof, Method('eof_action'))])