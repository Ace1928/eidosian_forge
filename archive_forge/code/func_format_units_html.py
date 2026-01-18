import re
def format_units_html(udict, font='%s', mult='&sdot;', paren=False):
    """
    Replace the units string provided with an equivalent html string.

    Exponentiation (m**2) will be replaced with superscripts (m<sup>2</sup>})

    No formatting is done, change `font` argument to e.g.:
    '<span style="color: #0000a0">%s</span>' to have text be colored blue.

    Multiplication (*) are replaced with the symbol specified by the mult
    argument. By default this is the latex &sdot; symbol.  Other useful options
    may be '' or '*'.

    If paren=True, encapsulate the string in '(' and ')'

    """
    from quantities.markup import format_units
    res = format_units(udict)
    if res.startswith('(') and res.endswith(')'):
        compound = True
    else:
        compound = False
    res = re.sub('\\*{2,2}(?P<exp>\\d+)', '<sup>\\g<exp></sup>', res)
    res = re.sub('\\*', mult, res)
    if paren and (not compound):
        res = '(%s)' % res
    res = font % res
    return res