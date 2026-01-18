import keyword
import os
import sys
import token
import tokenize
from IPython.utils.coloransi import TermColors, InputTermColors,ColorScheme, ColorSchemeTable
from .colorable import Colorable
from io import StringIO
def format2(self, raw, out=None):
    """ Parse and send the colored source.

        If out and scheme are not specified, the defaults (given to
        constructor) are used.

        out should be a file-type object. Optionally, out can be given as the
        string 'str' and the parser will automatically return the output in a
        string."""
    string_output = 0
    if out == 'str' or self.out == 'str' or isinstance(self.out, StringIO):
        out_old = self.out
        self.out = StringIO()
        string_output = 1
    elif out is not None:
        self.out = out
    else:
        raise ValueError('`out` or `self.out` should be file-like or the value `"str"`')
    if self.style == 'NoColor':
        error = False
        self.out.write(raw)
        if string_output:
            return (raw, error)
        return (None, error)
    colors = self.color_table[self.style].colors
    self.colors = colors
    self.raw = raw.expandtabs().rstrip()
    self.lines = [0, 0]
    pos = 0
    raw_find = self.raw.find
    lines_append = self.lines.append
    while True:
        pos = raw_find('\n', pos) + 1
        if not pos:
            break
        lines_append(pos)
    lines_append(len(self.raw))
    self.pos = 0
    text = StringIO(self.raw)
    error = False
    try:
        for atoken in generate_tokens(text.readline):
            self(*atoken)
    except tokenize.TokenError as ex:
        msg = ex.args[0]
        line = ex.args[1][0]
        self.out.write('%s\n\n*** ERROR: %s%s%s\n' % (colors[token.ERRORTOKEN], msg, self.raw[self.lines[line]:], colors.normal))
        error = True
    self.out.write(colors.normal + '\n')
    if string_output:
        output = self.out.getvalue()
        self.out = out_old
        return (output, error)
    return (None, error)