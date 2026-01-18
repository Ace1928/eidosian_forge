import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
def color_string(self):
    attrs = []
    if self.fg is not None:
        if self.fg in ansicolors:
            esc = codes[self.fg[5:]]
            if ';01m' in esc:
                self.bold = True
            attrs.append(esc[2:4])
        else:
            attrs.extend(('38', '5', '%i' % self.fg))
    if self.bg is not None:
        if self.bg in ansicolors:
            esc = codes[self.bg[5:]]
            attrs.append(str(int(esc[2:4]) + 10))
        else:
            attrs.extend(('48', '5', '%i' % self.bg))
    if self.bold:
        attrs.append('01')
    if self.underline:
        attrs.append('04')
    return self.escape(attrs)